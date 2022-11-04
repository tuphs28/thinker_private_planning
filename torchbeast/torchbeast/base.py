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
from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

from torchbeast.transformer_rnn import *
from torchbeast.train import *
from collections import deque

import gym
import gym_sokoban
import numpy as np

# yapf: disable

def define_parser():

    parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

    parser.add_argument("--env", type=str, default="Sokoban-v0",
                        help="Gym environment.")
    parser.add_argument("--env_disable_noop", action="store_true",
                        help="Disable noop in environment or not. (sokoban only)")

    parser.add_argument("--mode", default="train",
                        choices=["train", "test", "test_render"],
                        help="Training or test mode.")
    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")

    # Training settings.
    parser.add_argument("--disable_checkpoint", action="store_true",
                        help="Disable saving checkpoint.")
    parser.add_argument("--savedir", default="~/RS/thinker/logs/base",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--num_actors", default=32, type=int, metavar="N",
                        help="Number of actors (default: 32).")
    parser.add_argument("--total_steps", default=40000000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=32, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--unroll_length", default=20, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--num_buffers", default=None, type=int,
                        metavar="N", help="Number of shared-memory buffers.")
    parser.add_argument("--num_learner_threads", "--num_threads", default=1, type=int,
                        metavar="N", help="Number learner threads.")
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")


    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.005,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--model_cost", default=1,
                        type=float, help="Model cost/multiplier.")
    parser.add_argument("--reg_cost", default=1,
                        type=float, help="Reg cost/multiplier.")
    parser.add_argument("--discounting", default=0.97,
                        type=float, help="Discounting factor.")
    parser.add_argument("--lamb", default=0.97,
                        type=float, help="Lambda when computing trace.")
    parser.add_argument("--reward_clipping", default=10, type=int, 
                        metavar="N", help="Reward clipping.")
    parser.add_argument("--trun_bs", action="store_true",
                        help="Whether to add baseline as reward when truncated.")

    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.0004,
                        type=float, metavar="LR", help="Learning rate.")
    parser.add_argument("--disable_adam", action="store_true",
                        help="Use Aadm optimizer or not.")
    parser.add_argument("--alpha", default=0.99, type=float,
                        help="RMSProp smoothing constant.")
    parser.add_argument("--momentum", default=0, type=float,
                        help="RMSProp momentum.")
    parser.add_argument("--epsilon", default=0.01, type=float,
                        help="RMSProp epsilon.")
    parser.add_argument("--grad_norm_clipping", default=0.0, type=float,
                        help="Global gradient norm clip.")
    # yapf: enable

    return parser

parser = define_parser()

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

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
        env = environment.Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
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
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
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
    update_actor,
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

        if update_actor:
            pg_loss = compute_policy_gradient_loss(
                learner_outputs["policy_logits"],
                batch["action"],
                vtrace_returns.pg_advantages,
            )
        else:
            pg_loss = torch.zeros(1).to(learner_outputs["policy_logits"].device)

        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        if update_actor:
            entropy_loss = flags.entropy_cost * compute_entropy_loss(
                learner_outputs["policy_logits"]
            )
        else:
            entropy_loss = torch.zeros(1).to(learner_outputs["policy_logits"].device)
        reg_loss = flags.reg_cost * torch.sum(learner_outputs["reg_loss"])

        total_loss = pg_loss + baseline_loss + entropy_loss + reg_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "reg_loss": reg_loss.item()
        }

        optimizer.zero_grad()
        total_loss.backward()
        if flags.grad_norm_clipping > 0:
            nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
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
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        reg_loss=dict(size=(T + 1,), dtype=torch.float32)
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    mp.set_sharing_strategy('file_system')

    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    model = Net(env.observation_space.shape, env.action_space.n, flags)
    buffers = create_buffers(flags, env.observation_space.shape, model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context()
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(
        env.observation_space.shape, env.action_space.n, flags
    ).to(device=flags.device)

    if not flags.disable_adam:
        print("Using Adam...")        
        optimizer = torch.optim.Adam(
            learner_model.parameters(),
            lr=flags.learning_rate)
    else:
        print("Using RMS Prop...")
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha,
        )
    print("All parameters: ")
    for k, v in learner_model.named_parameters(): print(k, v.numel())    

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [         
        "mean_episode_return",
        "episode_returns",        
        "total_loss",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats, last_returns, tot_eps = 0, {}, deque(maxlen=400), 0

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats, last_returns, tot_eps
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler, step < flags.total_steps / 2
            )
            last_returns.extend(stats["episode_returns"])
            tot_eps = tot_eps + len(stats["episode_returns"])
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                to_log.update({"trail_mean_episode_return": np.average(last_returns) if len(last_returns) > 0 else 0.,
                               "episode": tot_eps})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))

            print_str =  "Steps %i @ %.1f SPS. Eps %i. L400 Return %f. Loss %f" % (step, sps, tot_eps, 
                np.average(last_returns) if len(last_returns) > 0 else 0., total_loss)

            for s in ["pg_loss", "baseline_loss", "entropy_loss", "reg_loss"]:
                if s in stats:
                    print_str += " %s %f" % (s, stats[s])

            logging.info(print_str)
    except KeyboardInterrupt:
        for thread in threads:
            thread.join()        
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def test(flags, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []
    core_state = model.initial_state(batch_size=1)

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        policy_outputs, core_state = model(observation, core_state)
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )

class BaseNet(nn.Module):
    def __init__(self, observation_shape, num_actions, flags):

        super(BaseNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions  
        self.conv_out_hw = 8 
        self.conv_out = 32
        
        self.conv1_p = nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=8, stride=4)        
        self.conv2_p = nn.Conv2d(32, self.conv_out, kernel_size=4, stride=2)    
        self.frame_conv_p = torch.nn.Sequential(self.conv1_p, nn.ReLU(), self.conv2_p, nn.ReLU())

        self.conv1_v = nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=8, stride=4)        
        self.conv2_v = nn.Conv2d(32, self.conv_out, kernel_size=4, stride=2)    
        self.frame_conv_v = torch.nn.Sequential(self.conv1_v, nn.ReLU(), self.conv2_v, nn.ReLU())

        self.env_input_size = self.conv_out        
        
        self.fc_p = nn.Linear(self.conv_out_hw * self.conv_out_hw * self.conv_out, 256)                
        self.fc_v = nn.Linear(self.conv_out_hw * self.conv_out_hw * self.conv_out, 256)        

        self.policy = nn.Linear(256, self.num_actions)        
        self.baseline = nn.Linear(256, 1)
        
        print("model size: ", sum(p.numel() for p in self.parameters()))

    def initial_state(self, batch_size):

        return ()

    def forward(self, inputs, core_state=(), debug=False):

        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0        
        
        env_input_p = self.frame_conv_p(x)   
        core_output_p = F.relu(self.fc_p(torch.flatten(env_input_p, start_dim=1)))           
        policy_logits = self.policy(core_output_p)

        env_input_v = self.frame_conv_v(x)   
        core_output_v = F.relu(self.fc_v(torch.flatten(env_input_v, start_dim=1)))           
        baseline = self.baseline(core_output_v)
        
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)        
        
        reg_loss = torch.zeros(T, B).to(x.device)    

        ret_dict = dict(policy_logits=policy_logits, baseline=baseline, 
                         action=action, reg_loss=reg_loss)
        if debug: 
            debug_dict = dict(env_input=env_input, 
                              core_output=output, 
                              last_layer=core_output.view(T, B, -1))
            ret_dict.update(debug_dict)
            for n, i in enumerate(aw):
                ret_dict["attn_%d"%n] = i

        return (ret_dict, core_state)  


Net = BaseNet

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


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
