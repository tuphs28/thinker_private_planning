# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from torchbeast import atari_wrappers
from torchbeast.core.environment import Environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

from torchbeast.transformer_rnn import *
from torchbeast.resnet import ResNet, BasicBlock
from collections import deque

import gym
import gym_sokoban
import numpy as np

# yapf: disable

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())

def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:                
                if key in buffers: buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                if key in buffers: buffers[key][index][0, ...] = agent_output[key]                    
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                if flags.trun_bs:
                    if env_output['truncated_done']: 
                        env_output['reward'] = env_output['reward'] + flags.discounting * agent_output['baseline']

                timings.time("step")

                for key in env_output:
                    if key in buffers:
                        buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    if key in buffers:
                        buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping > 0:
            clipped_rewards = torch.clamp(rewards, -flags.reward_clipping, flags.reward_clipping)
        else:
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
            lamb=flags.lamb
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )
        if flags.im_entropy_cost > 0:
            im_entropy_loss = (1 / flags.rec_t) * flags.im_entropy_cost * (
                compute_entropy_loss(torch.flatten(
                    learner_outputs["im_policy_logits"], start_dim=1, end_dim=2)))
        else:
            im_entropy_loss = torch.tensor(0., device=flags.device)

        if flags.reset_entropy_cost > 0:
            reset_entropy_loss = (1 / flags.rec_t) * flags.reset_entropy_cost * (
                compute_entropy_loss(torch.flatten(
                    learner_outputs["reset_policy_logits"], start_dim=1, end_dim=2)))
        else:
            reset_entropy_loss = torch.tensor(0., device=flags.device)

        reg_loss = flags.reg_cost * torch.sum(learner_outputs["reg_loss"])

        total_loss = pg_loss + baseline_loss + entropy_loss + im_entropy_loss + reset_entropy_loss + reg_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "im_entropy_loss": im_entropy_loss.item(),
            "reset_entropy_loss": reset_entropy_loss.item(),
            "reg_loss": reg_loss.item()
        }

        optimizer.zero_grad()
        total_loss.backward()
        optimize_params = optimizer.param_groups[0]['params']

        if flags.grad_norm_clipping > 0:
            total_norm = nn.utils.clip_grad_norm_(optimize_params, flags.grad_norm_clipping)
        else:
            total_norm = 0.
            parameters = [p for p in optimize_params if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        stats["total_norm"] = total_norm

        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats
            
    
def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        truncated_done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        im_policy_logits=dict(size=(T + 1, flags.rec_t, num_actions), dtype=torch.float32),
        reset_policy_logits=dict(size=(T + 1, flags.rec_t, 2), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),        
        baseline=dict(size=(T + 1,), dtype=torch.float32),        
        reg_loss=dict(size=(T + 1,), dtype=torch.float32),
        uniform=dict(size=(T + 1, flags.rec_t, num_actions+2), dtype=torch.float32)
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def create_env(flags):
    if flags.env == "Sokoban-v0":
        #print("noop enabled: ", not flags.env_disable_noop)
        return atari_wrappers.SokobanWrapper(gym.make("Sokoban-v0"), noop=not flags.env_disable_noop)

    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )

