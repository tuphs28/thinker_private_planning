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
from torch.multiprocessing import Process, Manager
from torch import nn
from torch.nn import functional as F

from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace
from torchbeast.core.environment import Environment, Vec_Environment
from torchbeast.atari_wrappers import *
from torchbeast.transformer_rnn import *
from torchbeast.train import *
from torchbeast.model import Model

import gym
import gym_sokoban
import numpy as np
from matplotlib import pyplot as plt
import logging
from collections import deque

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

torch.multiprocessing.set_sharing_strategy('file_system')

# Update to original core funct

def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        truncated_done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        im_policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        reset_policy_logits=dict(size=(T + 1, 2), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1, 3), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        im_action=dict(size=(T + 1,), dtype=torch.int64),
        reset_action=dict(size=(T + 1,), dtype=torch.int64),
        cur_t=dict(size=(T + 1,), dtype=torch.int64),
        reg_loss=dict(size=(T + 1,), dtype=torch.float32),        
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers  

def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    actor_net: torch.nn.Module,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = ModelWrapper(SokobanWrapper(gym.make("Sokoban-v0"), noop=not flags.env_disable_noop), 
                               model=model, rec_t=flags.rec_t, discounting=flags.discounting, 
                               aug_stat=flags.aug_stat, no_mem=flags.no_mem)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = Environment(gym_env)
        env_output = env.initial()
        agent_state = actor_net.initial_state(batch_size=1)
        agent_output, unused_state = actor_net(env_output, agent_state)
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
                    agent_output, agent_state = actor_net(env_output, agent_state)                    

                timings.time("actor_net")
                
                action = torch.cat([agent_output['action'], agent_output['im_action'], agent_output['reset_action']], dim=-1)
                env_output = env.step(action.unsqueeze(0))

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

def compute_policy_gradient_loss(logits, im_logits, reset_logits,
    actions, im_actions, reset_actions, cur_t, advantages):
    re_cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",)
    im_cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(im_logits, 0, 1), dim=-1),
        target=torch.flatten(im_actions, 0, 1),
        reduction="none",) 
    reset_cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(reset_logits, 0, 1), dim=-1),
        target=torch.flatten(reset_actions, 0, 1),
        reduction="none",)    
    re_mask = (cur_t == 0).view_as(re_cross_entropy)
    cross_entropy = torch.where(re_mask, re_cross_entropy, im_cross_entropy+reset_cross_entropy)
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())  

def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions) 
  
def from_logits(
    behavior_policy_logits,
    behavior_im_policy_logits,
    behavior_reset_policy_logits,
    target_policy_logits,
    target_im_policy_logits,
    target_reset_policy_logits,            
    actions,
    im_actions,
    reset_actions,
    cur_t,     
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    lamb=1.0,
):
    """V-trace for softmax policies."""
    
    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    target_im_reset_action_log_probs = (action_log_probs(target_im_policy_logits, im_actions) +
        action_log_probs(target_reset_policy_logits, reset_actions))
    
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    behavior_im_reset_action_log_probs = (action_log_probs(behavior_im_policy_logits, im_actions) +
        action_log_probs(behavior_reset_policy_logits, reset_actions))    
    
    log_rhos = torch.where(cur_t==0, target_action_log_probs - behavior_action_log_probs,
        target_im_reset_action_log_probs - behavior_im_reset_action_log_probs)    
    
    vtrace_returns = vtrace.from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        lamb=lamb
    )
    return vtrace.VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )  
  
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

        vtrace_returns = from_logits(
            behavior_policy_logits=batch["policy_logits"],
            behavior_im_policy_logits=batch["im_policy_logits"],
            behavior_reset_policy_logits=batch["reset_policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            target_im_policy_logits=learner_outputs["im_policy_logits"],
            target_reset_policy_logits=learner_outputs["reset_policy_logits"],            
            actions=batch["action"],
            im_actions=batch["im_action"],
            reset_actions=batch["reset_action"],
            cur_t=batch["cur_t"],          
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
            lamb=flags.lamb
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            learner_outputs["im_policy_logits"],
            learner_outputs["reset_policy_logits"],                
            batch["action"],
            batch["im_action"],
            batch["reset_action"],
            batch["cur_t"],      
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        re_entropy_loss = compute_entropy_loss(learner_outputs["policy_logits"])
        im_entropy_loss = compute_entropy_loss(learner_outputs["im_policy_logits"])                                       
        reset_entropy_loss = compute_entropy_loss(learner_outputs["reset_policy_logits"]) 
        entropy_loss = flags.entropy_cost * (re_entropy_loss)
        im_entropy_loss = flags.im_entropy_cost * (im_entropy_loss + reset_entropy_loss)
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
            "im_entropy_loss": im_entropy_loss.item(),
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

# Wrap the environment with a model

def _format_frame(frame, bsz=None):
    #frame = torch.from_numpy(frame)
    if bsz is not None:
        return frame.view((1,) + frame.shape)
    else:
        return frame.view((1, 1) + frame.shape)

class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        initial_frame = _format_frame(self.gym_env.reset())
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            truncated_done=torch.tensor(0).view(1, 1).bool(),
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            cur_t=torch.tensor(0).view(1, 1),
            last_action=initial_last_action,
        )

    def step(self, action):
        frame, reward, done, unused_info = self.gym_env.step(action[0,0].cpu().detach().numpy())     
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        truncated_done = 'TimeLimit.truncated' in unused_info and unused_info['TimeLimit.truncated']
        truncated_done = torch.tensor(truncated_done).view(1, 1)
        cur_t = torch.tensor(unused_info["cur_t"]).view(1, 1)
        return dict(
            frame=frame,
            reward=reward,
            done=done,
            truncated_done=truncated_done,          
            episode_return=episode_return,
            episode_step=episode_step,
            cur_t=cur_t,
            last_action=action,
        )

    def close(self):
        self.gym_env.close()

    def clone_state(self):
        state = [self.episode_return.clone(), self.episode_step.clone()]
        state.append(self.gym_env.clone_state())
        return state
        
    def restore_state(self, state):
        self.episode_return = state[0].clone()
        self.episode_step = state[1].clone()
        self.gym_env.restore_state(state[2])
        
class ModelWrapper(gym.Wrapper):
    def __init__(self, env, model, rec_t, discounting=0.99, aug_stat=True, no_mem=True):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.model = model                
        self.rec_t = rec_t
        self.num_actions = env.action_space.n
        obs_n = (7 + num_actions * 7 + self.rec_t if aug_stat else 
            5 + num_actions * 3 + self.rec_t)
        self.observation_space = gym.spaces.Box(
          low=-np.inf, high=np.inf, shape=(obs_n, 1, 1), dtype=float)
        self.discounting = discounting
        self.aug_stat = aug_stat
        self.use_model = self.use_model_aug if aug_stat else self.use_model_base
        self.no_mem = no_mem
        self.model.train(False)
        
    def reset(self, **kwargs):
        x = self.env.reset()
        self.cur_t = 0        
        out = self.use_model(x, 0., 0, self.cur_t, 1.)        
        return out.unsqueeze(-1).unsqueeze(-1)
    
    def step(self, action):  
        re_action, im_action, reset = action
        if self.cur_t < self.rec_t - 1:
          self.cur_t += 1
          out = self.use_model(None, None, im_action, self.cur_t, reset)          
          r = 0.
          done = False
          info = {'cur_t': self.cur_t}  
          self.encoded = reset * self.encoded_reset + (1 - reset) * self.encoded
        else:
          self.cur_t = 0
          x, r, done, info = self.env.step(re_action)          
          out = self.use_model(x, r, re_action, self.cur_t, 1.)          
          info['cur_t'] = self.cur_t
        return out.unsqueeze(-1).unsqueeze(-1), r, done, info
        
        
    def use_model_aug(self, x, r, a, cur_t, reset):
        # input: 
        # r: reward - [,]; x: frame - [C, H, W]; a: action - [,]
        # cur_t: int; reset at cur_t == 0  
        with torch.no_grad():
            if cur_t == 0:
                self.rollout_depth = 0.
                if self.no_mem:
                    self.re_action = F.one_hot(torch.zeros(1, dtype=torch.long), self.num_actions)   
                else:
                    self.re_action = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions)                   
                
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                _, vs, logits, encodeds = self.model(x, self.re_action.unsqueeze(0), one_hot=True)                
                self.encoded = encodeds[-1]    
                self.encoded_reset = self.encoded.clone()
                
                if self.no_mem:
                    self.re_reward = torch.tensor([[0.]], dtype=torch.float32)                
                else:
                    self.re_reward = torch.tensor([[r]], dtype=torch.float32)                
                    
                self.v0 = vs[-1].unsqueeze(-1).clone()
                self.logit0 = logits[-1].clone()
                
                self.im_action = torch.zeros(1, self.num_actions, dtype=torch.float32)
                self.im_reset = torch.tensor([[1.]], dtype=torch.float32)
                self.im_reward = torch.zeros(1, 1, dtype=torch.float32)                                
                self.v = vs[-1].unsqueeze(-1)
                self.logit = logits[-1]
                self.rollout_first_action = torch.zeros(1, self.num_actions, dtype=torch.float32)
                self.rollout_return_wo_v = torch.zeros(1, 1, dtype=torch.float32)   
                self.rollout_return = torch.zeros(1, 1, dtype=torch.float32)                
                self.q_s_a = torch.zeros(1, self.num_actions, dtype=torch.float32)
                self.n_s_a = torch.zeros(1, self.num_actions, dtype=torch.float32)                
            else:
                self.rollout_depth += 1                
                
                self.im_action = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions)   
                rs, vs, logits, encodeds = self.model.forward_encoded(self.encoded, 
                   self.im_action.unsqueeze(0), one_hot=True)
                self.encoded = encodeds[-1]        
                
                self.im_reward = rs[-1].unsqueeze(-1)
                self.v = vs[-1].unsqueeze(-1)    
                self.logit = logits[-1]     
                
                if self.im_reset: 
                    # last action's reset is true; re-initialize everything                    
                    self.rollout_first_action = self.im_action.clone()
                    self.rollout_return_wo_v = torch.zeros(1, 1, dtype=torch.float32)   
                    self.rollout_depth = 1                      
                    
                self.rollout_return_wo_v += (self.discounting ** (self.rollout_depth-1)) * self.im_reward
                self.rollout_return = self.rollout_return_wo_v + (
                    self.discounting ** (self.rollout_depth)) * self.v                    
                    
                self.im_reset = torch.tensor([[reset]], dtype=torch.float32)
                if self.im_reset:                    
                    rollout_first_action_label = torch.argmax(self.rollout_first_action, dim=1)                    
                    q = self.q_s_a[:, rollout_first_action_label]
                    n = self.n_s_a[:, rollout_first_action_label]                    
                    ret = self.rollout_return[:, 0]
                    self.n_s_a[:, rollout_first_action_label] += 1                    
                    self.q_s_a[:, rollout_first_action_label] = (n * q) / (n + 1) + ret / (n + 1)
        time = F.one_hot(torch.tensor([cur_t]).long(), self.rec_t)
        depc = torch.tensor([[self.discounting ** (self.rollout_depth-1)]])
        ret_dict = {"re_action": self.re_action,
                    "re_reward": self.re_reward,
                    "v0": self.v0,
                    "logit0": self.logit0,
                    "im_action": self.im_action,
                    "im_reset": self.im_reset,
                    "im_reward": self.im_reward,
                    "v": self.v,
                    "logit": self.logit,
                    "rollout_first_action": self.rollout_first_action,
                    "rollout_return": self.rollout_return,
                    "n_s_a": self.n_s_a,
                    "q_s_a": self.q_s_a,
                    "time": time,
                    "depc": depc}
        self.ret_dict = ret_dict
        out = torch.concat(list(ret_dict.values()), dim=-1)   
        return out[0]        
        
    def use_model_base(self, x, r, a, cur_t, reset):
        # input: 
        # r: reward - [,]; x: frame - [C, H, W]; a: action - [,]
        # cur_t: int; reset at cur_t == 0  
        a = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions)           
        reset = torch.tensor([[reset]], dtype=torch.float32)
        if cur_t == 0:
          x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
          r = torch.tensor(r, dtype=torch.float32).unsqueeze(0)    
          with torch.no_grad():
              rs, vs, logits, encodeds = self.model(x, a.unsqueeze(0), one_hot=True)
          self.encoded_reset = encodeds[0].clone()
        else:
          with torch.no_grad():
              rs, vs, logits, encodeds = self.model.forward_encoded(self.encoded, 
                   a.unsqueeze(0), one_hot=True)
          
        self.encoded = encodeds[-1]        
        r = (r if cur_t == 0 else rs[-1]).unsqueeze(-1)
        v = vs[-1].unsqueeze(-1)
        logit = logits[-1]    
        
        if cur_t == 0:
          self.r0 = r.clone()
          self.v0 = v.clone()
          self.logit0 = logit.clone()        
        time = F.one_hot(torch.tensor([cur_t], device=a.device).long(), self.rec_t)
        out = torch.concat([reset, a, r, v, logit, self.r0, self.v0, self.logit0, time], dim=-1)   
        return out[0]
        
class Actor_net(nn.Module):    
    def __init__(self, obs_shape, num_actions, flags):

        super(Actor_net, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions  
        
        self.tran_t = flags.tran_t                   # number of recurrence of RNN
        self.tran_mem_n = flags.tran_mem_n           # size of memory for the attn modules
        self.tran_layer_n = flags.tran_layer_n       # number of layers
        self.tran_lstm = flags.tran_lstm             # to use lstm or not
        self.tran_lstm_no_attn = flags.tran_lstm_no_attn  # to use attention in lstm or not
        self.tran_norm_first = flags.tran_norm_first # to use norm first in transformer (not on LSTM)
        self.tran_ff_n = flags.tran_ff_n             # number of dim of ff in transformer (not on LSTM)        
        self.tran_skip = flags.tran_skip             # whether to add skip connection
        self.conv_out = flags.tran_dim               # size of transformer / LSTM embedding dim
        self.no_mem = flags.no_mem
        
        self.conv_out_hw = 1   
        self.d_model = self.conv_out
        
        self.conv1 = nn.Conv2d(in_channels=self.obs_shape[0], out_channels=self.conv_out//2, kernel_size=1, stride=1)        
        self.conv2 = nn.Conv2d(in_channels=self.conv_out//2, out_channels=self.conv_out, kernel_size=1, stride=1)        
        self.frame_conv = torch.nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU())
        self.env_input_size = self.conv_out
        d_in = self.env_input_size + self.d_model 
        
        if self.tran_lstm:
            self.core = ConvAttnLSTM(h=self.conv_out_hw, w=self.conv_out_hw,
                                 input_dim=d_in-self.d_model, hidden_dim=self.d_model,
                                 kernel_size=1, num_layers=self.tran_layer_n,
                                 num_heads=8, mem_n=self.tran_mem_n, attn=not self.tran_lstm_no_attn)
        else:            
            self.core = ConvTransformerRNN(d_in=d_in,
                                       h=self.conv_out_hw, w=self.conv_out_hw, d_model=self.d_model, 
                                       num_heads=8, dim_feedforward=self.tran_ff_n, 
                                       mem_n=self.tran_mem_n, norm_first=self.tran_norm_first,
                                       num_layers=self.tran_layer_n, rpos=True, conv=False)   
                         
        
        if self.tran_skip:
            rnn_out_size = self.conv_out_hw * self.conv_out_hw * (self.d_model + self.env_input_size)
        else:
            rnn_out_size = self.conv_out_hw * self.conv_out_hw * self.d_model
                
        self.fc = nn.Linear(rnn_out_size, 256)        
        
        self.im_policy = nn.Linear(256, self.num_actions)        
        self.policy = nn.Linear(256, self.num_actions)        
        self.baseline = nn.Linear(256, 1)        
        self.reset = nn.Linear(256, 2)        
        
        print("actor size: ", sum(p.numel() for p in self.parameters()))
        #for k, v in self.named_parameters(): print(k, v.numel())   

    def initial_state(self, batch_size):
        state = self.core.init_state(batch_size) + (torch.zeros(1, batch_size, 
               self.env_input_size, self.conv_out_hw, self.conv_out_hw),)
        return state

    def forward(self, obs, core_state=(), debug=False):
        # one-step forward for the actor
        # input / done shape x: T x B x C x 1 x 1 / B x C x 1 x 1
        # only supports T = 1 at the moment; all output does not have T dim.        
        
        x = obs["frame"]
        done = obs["done"]
        
        if len(x.shape) == 4: x = x.unsqueeze(0)
        if len(done.shape) == 1: done = done.unsqueeze(0)  
            
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.  
        env_input = self.frame_conv(x)                
        core_input = env_input.view(T, B, -1, self.conv_out_hw, self.conv_out_hw)
        core_output_list = []
        notdone = ~(done.bool())
        
        for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):                
            if self.no_mem and obs["cur_t"][n, 0] == 0:
                core_state = self.initial_state(B)
                core_state = tuple(v.to(x.device) for v in core_state)
                
            # Input shape: B, self.conv_out + self.num_actions + 1, H, W
            for t in range(self.tran_t):                
                if t > 0: nd = torch.ones(B).to(x.device).bool()                    
                nd = nd.view(-1)      
                output, core_state = self.core(input, core_state, nd, nd) # output shape: 1, B, core_output_size 
                
            last_input = input   
            core_output_list.append(output)
                                   
        core_output = torch.cat(core_output_list)  
        if self.tran_skip: core_output = torch.concat([core_output, core_input], dim=-3)
        core_output = torch.flatten(core_output, 0, 1)        
        core_output = F.relu(self.fc(torch.flatten(core_output, start_dim=1)))   
        
        policy_logits = self.policy(core_output)
        im_policy_logits = self.im_policy(core_output)
        reset_policy_logits = self.reset(core_output)
        
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        im_action = torch.multinomial(F.softmax(im_policy_logits, dim=1), num_samples=1)
        reset_action = torch.multinomial(F.softmax(reset_policy_logits, dim=1), num_samples=1)
                
        baseline = self.baseline(core_output)
                   
        reg_loss = (1e-3 * torch.sum(policy_logits**2, dim=-1) / 2 + 
                    1e-5 * torch.sum(core_output**2, dim=-1) / 2)
        reg_loss = reg_loss.view(T, B)
        
        policy_logits = policy_logits.view(T, B, self.num_actions)
        im_policy_logits = im_policy_logits.view(T, B, self.num_actions)
        reset_policy_logits = reset_policy_logits.view(T, B, 2)
        
        action = action.view(T, B)      
        im_action = im_action.view(T, B)      
        reset_action = reset_action.view(T, B)             
        baseline = baseline.view(T, B)
        
        ret_dict = dict(policy_logits=policy_logits,                         
                        im_policy_logits=im_policy_logits,                         
                        reset_policy_logits=reset_policy_logits,     
                        action=action,     
                        im_action=im_action,
                        reset_action=reset_action,
                        baseline=baseline, 
                        reg_loss=reg_loss, )
        
        return (ret_dict, core_state)      

def define_parser():

    parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

    parser.add_argument("--env", type=str, default="Sokoban-v0",
                        help="Gym environment.")
    parser.add_argument("--env_disable_noop", action="store_true",
                        help="Disable noop in environment or not. (sokoban only)")

    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")

    # Training settings.
    parser.add_argument("--disable_checkpoint", action="store_true",
                        help="Disable saving checkpoint.")
    parser.add_argument("--savedir", default="~/RS/thinker/logs/torchbeast",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--num_actors", default=48, type=int, metavar="N",
                        help="Number of actors (default: 48).")
    parser.add_argument("--total_steps", default=500000000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=32, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--unroll_length", default=100, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--num_buffers", default=None, type=int,
                        metavar="N", help="Number of shared-memory buffers.")
    parser.add_argument("--num_learner_threads", "--num_threads", default=1, type=int,
                        metavar="N", help="Number learner threads.")
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")

    # Architecture settings
    parser.add_argument("--tran_dim", default=64, type=int, metavar="N",
                        help="Size of transformer hidden dim.")
    parser.add_argument("--tran_mem_n", default=5, type=int, metavar="N",
                        help="Size of transformer memory.")
    parser.add_argument("--tran_layer_n", default=3, type=int, metavar="N",
                        help="Number of transformer layer.")
    parser.add_argument("--tran_t", default=1, type=int, metavar="T",
                        help="Number of recurrent step for transformer.")
    parser.add_argument("--tran_ff_n", default=256, type=int, metavar="N",
                        help="Size of transformer ff .")
    parser.add_argument("--tran_skip", action="store_true",
                        help="Whether to enable skip conn.")
    parser.add_argument("--tran_norm_first", action="store_true",
                        help="Whether to use norm first in transformer.")
    parser.add_argument("--tran_rpos", action="store_true",
                        help="Whether to use relative position in transformer.")
    parser.add_argument("--tran_lstm", action="store_true",
                        help="Whether to use LSTM-transformer.")
    parser.add_argument("--tran_lstm_no_attn", action="store_true",
                        help="Whether to disable attention in LSTM-transformer.")
    parser.add_argument("--tran_erasep", action="store_true",
                        help="Whether to erase past memories if not planning.")

    parser.add_argument("--rec_t", default=5, type=int, metavar="N",
                        help="Number of planning steps.")
    parser.add_argument("--aug_stat", action="store_true",
                        help="Whether to use augmented stat.")    
    parser.add_argument("--no_mem", action="store_true",
                        help="Whether to erase all memories after each real action.")   
    
    parser.add_argument("--model_type_nn", default=0,
                        type=float, help="Model type.")      
    
    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.0001,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--im_entropy_cost", default=0.000001,
                        type=float, help="Imagainary Entropy cost/multiplier.")    
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--reg_cost", default=0.1,
                        type=float, help="Reg cost/multiplier.")
    parser.add_argument("--discounting", default=0.99,
                        type=float, help="Discounting factor.")
    parser.add_argument("--lamb", default=1.,
                        type=float, help="Lambda when computing trace.")
    parser.add_argument("--reward_clipping", default=10, type=int, 
                        metavar="N", help="Reward clipping.")
    parser.add_argument("--trun_bs", action="store_true",
                        help="Whether to add baseline as reward when truncated.")

    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.00005,
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
flags = parser.parse_args()        

raw_env = SokobanWrapper(gym.make("Sokoban-v0"), noop=not flags.env_disable_noop)
raw_obs_shape, num_actions = raw_env.observation_space.shape, raw_env.action_space.n 

model = Model(flags, raw_obs_shape, num_actions=num_actions)
checkpoint = torch.load("../models/model_1.tar")
model.load_state_dict(checkpoint["model_state_dict"])    

env = Environment(ModelWrapper(SokobanWrapper(gym.make("Sokoban-v0"), noop=not flags.env_disable_noop), 
     model=model, rec_t=flags.rec_t, discounting=flags.discounting, aug_stat=flags.aug_stat, no_mem=flags.no_mem))

obs_shape = env.gym_env.observation_space.shape

logging.basicConfig(format='%(message)s', level=logging.DEBUG)

mp.set_sharing_strategy('file_system')

if flags.xpid is None:
    flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
plogger = file_writer.FileWriter(
    xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
)

flags.device = None
if not flags.disable_cuda and torch.cuda.is_available():
    logging.info("Using CUDA.")
    flags.device = torch.device("cuda")
else:
    logging.info("Not using CUDA.")
    flags.device = torch.device("cpu")

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

actor_net = Actor_net(obs_shape, num_actions, flags)
buffers = create_buffers(flags, obs_shape, num_actions)

actor_net.share_memory()

# Add initial RNN state.
initial_agent_state_buffers = []
for _ in range(flags.num_buffers):
    state = actor_net.initial_state(batch_size=1)
    for t in state:
        t.share_memory_()
    initial_agent_state_buffers.append(state)

actor_processes = []
ctx = mp.get_context()
free_queue = ctx.SimpleQueue()
full_queue = ctx.SimpleQueue()

for i in range(flags.num_actors):
    actor = ctx.Process(target=act, args=(flags, i, free_queue, full_queue,
            actor_net, model, buffers, initial_agent_state_buffers,),)
    actor.start()
    actor_processes.append(actor)

learner_net = Actor_net(obs_shape, num_actions, flags).to(device=flags.device)

if not flags.disable_adam:
    print("Using Adam...")        
    optimizer = torch.optim.Adam(learner_net.parameters(),lr=flags.learning_rate)
else:
    print("Using RMS Prop...")
    optimizer = torch.optim.RMSprop(
        learner_net.actor.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,)
print("All parameters: ")
for k, v in learner_net.named_parameters(): print(k, v.numel())    

def lr_lambda(epoch):
    return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

logger = logging.getLogger("logfile")
stat_keys = ["mean_episode_return", "episode_returns", "total_loss",
    "pg_loss", "baseline_loss", "entropy_loss", "im_entropy_loss"]
logger.info("# Step\t%s", "\t".join(stat_keys))

step, stats, last_returns, tot_eps = 0, {}, deque(maxlen=400), 0

def batch_and_learn(i, lock=threading.Lock()):
    """Thread target for the learning process."""
    #nonlocal step, stats, last_returns, tot_eps
    global step, stats, last_returns, tot_eps
    timings = prof.Timings()
    while step < flags.total_steps:
        timings.reset()
        batch, agent_state = get_batch(flags, free_queue, full_queue, buffers,
            initial_agent_state_buffers, timings,)
        stats = learn(flags, actor_net, learner_net, batch, agent_state, optimizer, 
            scheduler)
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
            "model_state_dict": actor_net.state_dict(),
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

        for s in ["pg_loss", "baseline_loss", "entropy_loss", "im_entropy_loss", "reg_loss", "total_norm"]:
            if s in stats:
                print_str += " %s %f" % (s, stats[s])

        logging.info(print_str)
except KeyboardInterrupt:
    for thread in threads:
        thread.join()        
    # Try joining actors then quit.
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
