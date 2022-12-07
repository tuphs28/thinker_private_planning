import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True

import argparse
import logging
import os
import subprocess
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
from torchbeast.atari_wrappers import *
from torchbeast.transformer_rnn import *
from torchbeast.train import *
from torchbeast.model import Model

import gym
import gym_sokoban
import gym_csokoban
import numpy as np
from matplotlib import pyplot as plt
import logging
from collections import deque

torch.multiprocessing.set_sharing_strategy('file_system')

# util functions

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def exp_scale(x, start, end, n, m):
    a = (end - start) / (np.exp(m * n) - 1)
    c = start - a
    x = np.clip(x, 0, n)    
    return a * np.exp(m * x) + c

class DataParallelWrapper(object):
    def __init__(self, module):
        self.module = module
              
    def __getattr__(self, name):        
        if name in self.module.__dict__.keys(): 
            return getattr(self.module, name)
        else:
            return getattr(self.module.module, name)
    
# Update to original core funct

def create_buffers(flags, obs_shape, num_actions, num_rewards) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.float32),
        reward=dict(size=(T + 1, num_rewards), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        truncated_done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1, num_rewards), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        im_policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),        
        reset_policy_logits=dict(size=(T + 1, 2), dtype=torch.float32),
        baseline=dict(size=(T + 1, num_rewards), dtype=torch.float32),
        last_action=dict(size=(T + 1, 3 if not flags.flex_t else 4), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        im_action=dict(size=(T + 1,), dtype=torch.int64),        
        reset_action=dict(size=(T + 1,), dtype=torch.int64), 
        reg_loss=dict(size=(T + 1,), dtype=torch.float32),  
        cur_t=dict(size=(T + 1,), dtype=torch.int64),             
        max_rollout_depth=dict(size=(T + 1,), dtype=torch.float32),  
    )
    if flags.flex_t:
        specs.update(dict(
            term_policy_logits=dict(size=(T + 1, 2), dtype=torch.float32),
            term_action=dict(size=(T + 1,), dtype=torch.int64),)
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

        gym_env = ModelWrapper(EnvWrapper(gym.make(flags.env), noop=not flags.env_disable_noop, name=flags.env), 
                               model=model, flags=flags)
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
                
                action = [agent_output['action'], agent_output['im_action'], agent_output['reset_action']]
                if 'term_action' in agent_output: action.append(agent_output['term_action'])
                action = torch.cat(action, dim=-1)
                env_output = env.step(action.unsqueeze(0))

                if flags.trun_bs:
                    if env_output['truncated_done']: 
                        env_output['reward'] = env_output['reward'] + flags.im_discounting * agent_output['baseline']

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

def compute_baseline_loss(advantages, masks_ls, c_ls):
    assert len(masks_ls) == len(c_ls)
    loss = 0.  
    for mask, c in zip(masks_ls, c_ls):
        loss = loss + 0.5 * torch.sum((advantages * (1 - mask)) ** 2) * c        
    return loss
    
def compute_policy_gradient_loss(logits_ls, actions_ls, masks_ls, c_ls, advantages):
    assert len(logits_ls) == len(actions_ls) == len(masks_ls) == len(c_ls)
    loss = 0.    
    for logits, actions, masks, c in zip(logits_ls, actions_ls, masks_ls, c_ls):
        cross_entropy = F.nll_loss(
            F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
            target=torch.flatten(actions, 0, 1),
            reduction="none",)
        cross_entropy = cross_entropy.view_as(advantages)
        adv_cross_entropy = cross_entropy * advantages.detach()
        loss = loss + torch.sum(adv_cross_entropy * (1-masks)) * c
    return loss  

def compute_entropy_loss(logits_ls, masks_ls, c_ls):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    loss = 0.
    assert(len(logits_ls) == len(masks_ls) == len(c_ls))
    for logits, masks, c in zip(logits_ls, masks_ls, c_ls):
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        ent = torch.sum(policy * log_policy, dim=-1) #* (1-masks)
        loss = loss + torch.sum(ent) * c 
    return loss

def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions) 
  
def from_logits(
    behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
    discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0, lamb=1.0,):
    """V-trace for softmax policies."""
    assert(len(behavior_logits_ls) == len(target_logits_ls) == len(actions_ls) == len(masks_ls))
    log_rhos = 0.       
    for behavior_logits, target_logits, actions, masks in zip(behavior_logits_ls, 
             target_logits_ls, actions_ls, masks_ls):
        behavior_log_probs = action_log_probs(behavior_logits, actions)        
        target_log_probs = action_log_probs(target_logits, actions)
        log_rho = target_log_probs - behavior_log_probs
        log_rhos = log_rhos + log_rho * (1-masks)
    
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
        behavior_action_log_probs=None,
        target_action_log_probs=None,
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
    real_step,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    
    with lock:                
        learner_outputs, unused_state = model(batch, initial_agent_state)
        #learner_outputs["im_policy_logits"].register_hook(lambda grad: grad / (flags.rec_t - 1))
        #learner_outputs["reset_policy_logits"].register_hook(lambda grad: grad / (flags.rec_t - 1))
        #learner_outputs["baseline"].register_hook(lambda grad: grad / (flags.rec_t - 1))
        
        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]        

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}
        
        T, B = batch["done"].shape

        rewards = batch["reward"]
        if flags.reward_clipping > 0:
            clipped_rewards = torch.clamp(rewards, -flags.reward_clipping, flags.reward_clipping)
        else:
            clipped_rewards = rewards
        
        # compute advantage w.r.t real rewards
        
        discounts = (~batch["done"]).float() * flags.im_discounting        
        #discounts = (~batch["done"]).float()
        #discounts[batch["cur_t"] == 0] = flags.discounting
        
        behavior_logits_ls = [batch["policy_logits"], batch["im_policy_logits"], batch["reset_policy_logits"]]
        target_logits_ls = [learner_outputs["policy_logits"], learner_outputs["im_policy_logits"], learner_outputs["reset_policy_logits"]]
        actions_ls = [batch["action"], batch["im_action"], batch["reset_action"]]        
        im_mask = (batch["cur_t"] == 0).float()
        real_mask = 1 - im_mask
        zero_mask = torch.zeros_like(im_mask)
        masks_ls = [real_mask, im_mask, im_mask]                
        c_ls = [flags.real_cost, flags.real_im_cost, flags.real_im_cost]
           
        if flags.flex_t:
            behavior_logits_ls.append(batch["term_policy_logits"])
            target_logits_ls.append(learner_outputs["term_policy_logits"])
            actions_ls.append(batch["term_action"])
            masks_ls.append(zero_mask)
            c_ls.append(flags.real_im_cost)

        vtrace_returns = from_logits(
            behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
            discounts=discounts,
            rewards=clipped_rewards[:, :, 0],
            values=learner_outputs["baseline"][:, :, 0],
            bootstrap_value=bootstrap_value[:, 0],
            lamb=flags.lamb
        )        
        
        pg_loss = compute_policy_gradient_loss(target_logits_ls, actions_ls, masks_ls, c_ls, vtrace_returns.pg_advantages, )  
        
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"][:, :, 0], 
            masks_ls = [real_mask, im_mask], c_ls = [flags.real_cost, flags.real_im_cost])
       
        # compute advantage w.r.t imagainary rewards

        if flags.reward_type == 1:
            if flags.reward_carry:                
                discounts = (~batch["done"]).float() * flags.im_discounting 
            else:
                discounts = (~(batch["cur_t"] == 0)).float() * flags.im_discounting        
            behavior_logits_ls = [batch["im_policy_logits"], batch["reset_policy_logits"]]
            target_logits_ls = [learner_outputs["im_policy_logits"], learner_outputs["reset_policy_logits"]]
            actions_ls = [batch["im_action"], batch["reset_action"]] 
            masks_ls = [im_mask, im_mask]  
            c_ls = [flags.im_cost, flags.im_cost]
            
            if flags.flex_t:
                behavior_logits_ls.append(batch["term_policy_logits"])
                target_logits_ls.append(learner_outputs["term_policy_logits"])
                actions_ls.append(batch["term_action"])
                masks_ls.append(zero_mask)
                c_ls.append(flags.im_cost)
            
            vtrace_returns = from_logits(
                behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
                discounts=discounts,
                rewards=clipped_rewards[:, :, 1],
                values=learner_outputs["baseline"][:, :, 1],
                bootstrap_value=bootstrap_value[:, 1],
                lamb=flags.lamb
            )
            im_pg_loss = compute_policy_gradient_loss(target_logits_ls, actions_ls, masks_ls, c_ls, vtrace_returns.pg_advantages, )   
            im_baseline_loss = flags.baseline_cost * compute_baseline_loss(
                vtrace_returns.vs - learner_outputs["baseline"][:, :, 1], masks_ls = [zero_mask], c_ls = [flags.im_cost])     
            
        target_logits_ls = [learner_outputs["policy_logits"], learner_outputs["im_policy_logits"], learner_outputs["reset_policy_logits"]]
        masks_ls = [real_mask, im_mask, im_mask]    
        im_ent_c = flags.im_entropy_cost * (flags.real_im_cost + (flags.im_cost if flags.reward_type == 1 else 0))
        c_ls = [flags.entropy_cost * flags.real_cost, im_ent_c, im_ent_c]
        if flags.flex_t:
            target_logits_ls.append(learner_outputs["term_policy_logits"])
            masks_ls.append(zero_mask)
            c_ls.append(im_ent_c)        
        entropy_loss = compute_entropy_loss(target_logits_ls, masks_ls, c_ls)       
            

        reg_loss = flags.reg_cost * torch.sum(learner_outputs["reg_loss"])
        total_loss = pg_loss + baseline_loss + entropy_loss + reg_loss         
              
        if flags.reward_type == 1:
            total_loss = total_loss + im_pg_loss + im_baseline_loss
        
        episode_returns = batch["episode_return"][batch["done"]][:, 0]  
        max_rollout_depth = (batch["max_rollout_depth"][batch["cur_t"] == 0]).detach().cpu().numpy()
        max_rollout_depth = np.average(max_rollout_depth) if len (max_rollout_depth) > 0 else 0.        
        real_step = torch.sum(batch["cur_t"]==0).item()
        stats = {
            "episode_returns": tuple(episode_returns.detach().cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "reg_loss": reg_loss.item(),
            "max_rollout_depth": max_rollout_depth,
            "real_step": real_step,
            "mean_plan_step": T * B / max(real_step, 1),
        }
        
        if flags.reward_type == 1:            
            im_episode_returns = batch["episode_return"][batch["cur_t"] == 0][:, 1]
            stats["im_episode_returns"] = tuple(im_episode_returns.detach().cpu().numpy())
            stats["im_pg_loss"] = im_pg_loss.item()
            stats["im_baseline_loss"] = im_baseline_loss.item()   

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
        if not flags.flex_t:
            scheduler.step()
        else:
            scheduler.last_epoch = real_step - 1  # scheduler does not support setting epoch directly
            scheduler.step() 

        actor_model.load_state_dict(model.state_dict())
        return stats  

# Wrap the environment with a model

def _format_frame(frame, bsz=None):
    if type(frame) == np.ndarray:
        frame = torch.from_numpy(frame).float()
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
        self.episode_return = torch.zeros(1, 1, 1)
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
        self.episode_return = self.episode_return + torch.tensor(reward).unsqueeze(0).unsqueeze(0)
        episode_step = self.episode_step
        episode_return = self.episode_return.clone()
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)        
        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1, -1)
        done = torch.tensor(done).view(1, 1)
        truncated_done = 'TimeLimit.truncated' in unused_info and unused_info['TimeLimit.truncated']
        truncated_done = torch.tensor(truncated_done).view(1, 1)
        cur_t = torch.tensor(unused_info["cur_t"]).view(1, 1)
        if cur_t == 0 and self.episode_return.shape[2] > 1:
            self.episode_return[:, :, 1] = 0.
        if 'max_rollout_depth' in unused_info:
            max_rollout_depth = torch.tensor(unused_info["max_rollout_depth"]).view(1, 1)
        else:
            max_rollout_depth = torch.tensor(0.).view(1, 1)
        
        return dict(
            frame=frame,
            reward=reward,
            done=done,
            truncated_done=truncated_done,          
            episode_return=episode_return,
            episode_step=episode_step,
            cur_t=cur_t,
            last_action=action,
            max_rollout_depth=max_rollout_depth
        )

    def close(self):
        self.gym_env.close()

    def clone_state(self):
        state = self.gym_env.clone_state()
        state["env_episode_return"] = self.episode_return.clone()
        state["env_episode_step"] = self.episode_step.clone()
        return state
        
    def restore_state(self, state):
        self.episode_return = state["env_episode_return"].clone()
        self.episode_step = state["env_episode_step"].clone()
        self.gym_env.restore_state(state)
        
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
        self.tran_lstm_new = flags.tran_lstm_new
        self.attn_mask_b = flags.tran_attn_b         # atention bias for current position
        self.tran_norm_first = flags.tran_norm_first # to use norm first in transformer (not on LSTM)
        self.tran_ff_n = flags.tran_ff_n             # number of dim of ff in transformer (not on LSTM)        
        self.tran_skip = flags.tran_skip             # whether to add skip connection
        self.conv_out = flags.tran_dim               # size of transformer / LSTM embedding dim        
        self.no_mem = flags.no_mem                   # whether to earse real memory at the end of planning stage
        self.num_rewards = flags.num_rewards         # dim of rewards (1 for vanilla; 2 for planning rewards)
        self.flex_t = flags.flex_t                   # whether to output the terminate action
        self.flex_t_term_b = flags.flex_t_term_b     # bias added to the logit of terminating
        
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
                                 num_heads=8, mem_n=self.tran_mem_n, attn=not self.tran_lstm_no_attn,
                                 attn_mask_b=self.attn_mask_b, legacy= not self.tran_lstm_new)
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
        self.baseline = nn.Linear(256, self.num_rewards)        
        self.reset = nn.Linear(256, 2)        
        
        if self.flex_t: self.term = nn.Linear(256, 2)        
        
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
        
        if self.flex_t: 
            term_policy_logits = self.term(core_output)            
            term_policy_logits[:, 1] += self.flex_t_term_b
        
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        im_action = torch.multinomial(F.softmax(im_policy_logits, dim=1), num_samples=1)
        reset_action = torch.multinomial(F.softmax(reset_policy_logits, dim=1), num_samples=1)
        if self.flex_t: term_action = torch.multinomial(F.softmax(term_policy_logits, dim=1), num_samples=1)
                
        baseline = self.baseline(core_output)
                   
        reg_loss = (1e-3 * torch.sum(policy_logits**2, dim=-1) / 2 + 
                    1e-5 * torch.sum(core_output**2, dim=-1) / 2)
        reg_loss = reg_loss.view(T, B)
        
        policy_logits = policy_logits.view(T, B, self.num_actions)
        im_policy_logits = im_policy_logits.view(T, B, self.num_actions)
        reset_policy_logits = reset_policy_logits.view(T, B, 2)
        if self.flex_t: term_policy_logits = term_policy_logits.view(T, B, 2)
            
        
        action = action.view(T, B)      
        im_action = im_action.view(T, B)      
        reset_action = reset_action.view(T, B)             
        if self.flex_t: term_action = term_action.view(T, B)
        baseline = baseline.view(T, B, self.num_rewards)
        
        ret_dict = dict(policy_logits=policy_logits,                         
                        im_policy_logits=im_policy_logits,                         
                        reset_policy_logits=reset_policy_logits,     
                        action=action,     
                        im_action=im_action,
                        reset_action=reset_action,
                        baseline=baseline, 
                        reg_loss=reg_loss, )
        
        if self.flex_t: ret_dict.update(dict(term_policy_logits=term_policy_logits,    
                                             term_action=term_action))
        return (ret_dict, core_state) 
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=200):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, step) :
        # step: int Tensor, shape [batch_size]
        step = torch.clamp(step, 0, self.max_len-1)
        return self.pe[step, :]    
        
class ModelWrapper(gym.Wrapper):
    def __init__(self, env, model, flags):
        gym.Wrapper.__init__(self, env)
        
        self.env = env
        self.model = model                
        self.rec_t = flags.rec_t        
        self.flex_t = flags.flex_t 
        self.flex_t_cost = flags.flex_t_cost         
        self.flex_t_cost_m = flags.flex_t_cost_m
        self.flex_t_cost_type = flags.flex_t_cost_type
        self.discounting = flags.discounting
        self.stat_pos_encode = flags.stat_pos_encode
        self.stat_pos_encode_dim = flags.stat_pos_encode_dim
        self.reward_type = flags.reward_type    
        self.no_mem = flags.no_mem
        self.perfect_model = flags.perfect_model
        self.reset_m = flags.reset_m
        self.tree_carry = flags.tree_carry
        self.tree_vb = flags.tree_vb
        self.thres_carry = flags.thres_carry        
        self.thres_discounting = flags.thres_discounting
        self.num_actions = env.action_space.n
        self.root_node = None
            
        if not self.flex_t:
            obs_n = 9 + num_actions * 10 + (self.rec_t if not self.stat_pos_encode else (2 * self.stat_pos_encode_dim))
        else:
            obs_n = 10 + num_actions * 10  + (1 if not self.stat_pos_encode else (2 * self.stat_pos_encode_dim))          
        if self.stat_pos_encode:
            obs_n = obs_n - num_actions * 2 + self.stat_pos_encode_dim * num_actions * 2
        
        self.observation_space = gym.spaces.Box(
          low=-np.inf, high=np.inf, shape=(obs_n, 1, 1), dtype=float)
        self.model.train(False)        
        
        self.max_rollout_depth = 0.
        self.thres = None
        self.root_max_q = None
        
        if self.stat_pos_encode:
            self.pos = PositionalEncoding(dim=self.stat_pos_encode_dim, max_len=self.rec_t)
        else:
            self.pos = None
        
    def reset(self, **kwargs):
        x = self.env.reset(**kwargs)
        self.cur_t = 0    
        out = self.use_model(x, 0., 0, self.cur_t, reset=1., term=0., done=False)
        if self.reward_type == 1:
            self.last_root_max_q = self.root_max_q
        return out.unsqueeze(-1).unsqueeze(-1)
    
    def step(self, action):  
        if not self.flex_t:
            re_action, im_action, reset = action
            term = None
        else:
            re_action, im_action, reset, term = action
        info = {}
        info["max_rollout_depth"] = self.max_rollout_depth
        if (not self.flex_t and self.cur_t < self.rec_t - 1) or (
            self.flex_t and self.cur_t < self.rec_t - 1 and not term):
          self.cur_t += 1
          out = self.use_model(None, None, im_action, self.cur_t, reset=reset, term=term, done=False)          
          if self.reward_type == 0:
            r = np.array([0.])
          else:
            if self.flex_t:
                if self.flex_t_cost_type == 0:
                    flex_t_cost = self.flex_t_cost
                elif self.flex_t_cost_type == 1:
                    flex_t_cost = exp_scale(self.cur_t, 1e-7, self.flex_t_cost, self.rec_t, self.flex_t_cost_m)
            else:                
                flex_t_cost = 0.
            r = np.array([0., (self.root_max_q - self.last_root_max_q - flex_t_cost).item()], dtype=np.float32)
          done = False
          info['cur_t'] = self.cur_t   
        else:
          self.cur_t = 0
          if self.perfect_model: self.env.restore_state(self.root_node.encoded)
          x, r, done, info_ = self.env.step(re_action)                    
          out = self.use_model(x, r, re_action, self.cur_t, reset=1., term=term, done=done) 
          info.update(info_)
          info['cur_t'] = self.cur_t
          if self.reward_type == 0:
            r = np.array([r])
          else:
            r = np.array([r, 0.], dtype=np.float32)   
        if self.reward_type == 1:
            self.last_root_max_q = self.root_max_q   
        
        return out.unsqueeze(-1).unsqueeze(-1), r, done, info        
        
    def use_model(self, x, r, a, cur_t, reset, term, done=False):     
        with torch.no_grad():
            if cur_t == 0:
                self.rollout_depth = 0.
                self.unexpand_rollout_depth = 0.
                self.pass_unexpand = False
                self.max_rollout_depth = 0.
                
                if self.root_max_q is not None:
                    self.thres = (self.root_max_q - r) / self.discounting
                if done:
                    self.thres = None
                
                if self.no_mem:
                    re_action = 0
                    re_reward = torch.tensor([0.], dtype=torch.float32)                
                else:
                    re_action = a                
                    re_reward = torch.tensor([r], dtype=torch.float32)                
                
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                self.x = self.x_ = x_tensor
                a_tensor = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions)                
                _, vs, logits, encodeds = self.model(x_tensor, a_tensor.unsqueeze(0), one_hot=True) 
                
                if self.perfect_model: 
                    encoded = self.clone_state()
                else:
                    encoded=encodeds[-1]
                
                if (not self.tree_carry or self.root_node is None or 
                    not self.root_node.children[a].expanded() or done):
                
                    self.root_node = Node(parent=None, action=re_action, logit=None, 
                                          num_actions=self.num_actions,
                                          discounting=self.discounting,
                                          rec_t=self.rec_t)
                    self.root_node.expand(r=torch.tensor([0.], dtype=torch.float32), 
                                          v=vs[-1, 0].unsqueeze(-1), logits=logits[-1, 0],
                                          encoded=encoded)
                else:
                    self.root_node = self.root_node.children[a]
                    self.root_node.expand(r=torch.tensor([0.], dtype=torch.float32), 
                                          v=vs[-1, 0].unsqueeze(-1), logits=logits[-1, 0],
                                          encoded=encoded, override=True)
                    self.parent = None
                
                if self.thres is not None:
                    self.thres = self.thres_discounting * self.thres + (1 - self.thres_discounting) * vs[-1, 0].item()
                
                self.root_node.visit()
                self.cur_node = self.root_node
                
            else:
                self.rollout_depth += 1                    
                self.max_rollout_depth = max(self.max_rollout_depth, self.rollout_depth)
                next_node = self.cur_node.children[a]
                
                if not next_node.expanded():
                    self.pass_unexpand = True
                    a_tensor = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions) 
                    if not self.perfect_model:
                        rs, vs, logits, encodeds = self.model.forward_encoded(self.cur_node.encoded, 
                            a_tensor.unsqueeze(0), one_hot=True)
                        next_node.expand(r=rs[-1, 0].unsqueeze(-1), v=vs[-1, 0].unsqueeze(-1), 
                                     logits=logits[-1, 0], encoded=encodeds[-1])
                    else:                        
                        if "done" not in self.cur_node.encoded:                            
                            self.env.restore_state(self.cur_node.encoded)                        
                            x, r, done, info = self.env.step(a) 
                            encoded = self.env.clone_state()
                            if done: encoded["done"] = True                        
                            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                            self.x_ = x_tensor
                            a_tensor = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions) 
                            _, vs, logits, _ = self.model(x_tensor, a_tensor.unsqueeze(0), one_hot=True)                        

                            if done:
                                v = torch.tensor([0.], dtype=torch.float32)
                            else:
                                v = vs[-1, 0].unsqueeze(-1)

                            next_node.expand(r=torch.tensor([r], dtype=torch.float32), 
                                             v=v, 
                                             logits=logits[-1, 0], 
                                             encoded=encoded)
                        else:
                            logits = torch.concat([x.logit for x in self.cur_node.children])  
                            next_node.expand(r=torch.tensor([0.], dtype=torch.float32), 
                                             v=torch.tensor([0.], dtype=torch.float32),
                                             logits=logits, 
                                             encoded=self.cur_node.encoded)                            
                            
                next_node.visit()
                self.cur_node = next_node
            
            if self.pass_unexpand:                 
                self.unexpand_rollout_depth += 1    
                if self.reset_m >= 0 and self.unexpand_rollout_depth > self.reset_m:
                    reset = True
            
            root_node_stat = self.root_node.stat(pos=self.pos)
            cur_node_stat = self.cur_node.stat(pos=self.pos)                        
            reset = torch.tensor([reset], dtype=torch.float32)            
            depc = torch.tensor([self.discounting ** (self.rollout_depth-1)])
            
            root_trail_r = self.root_node.trail_r / self.discounting
            root_rollout_q = self.root_node.rollout_q / self.discounting
            if self.tree_vb != 0:
                rollout_qs = [x + (self.tree_vb if n == 0 else 0.) for n, x in enumerate(self.root_node.rollout_qs)]
            else:
                rollout_qs = self.root_node.rollout_qs
            root_max_q = torch.max(torch.concat(rollout_qs)).unsqueeze(-1) / self.discounting
            if self.thres_carry and self.thres is not None:
                root_max_q = torch.max(root_max_q, self.thres)
                
            if self.stat_pos_encode:
                time = torch.concat([self.pos(torch.tensor([cur_t]).long()), self.pos(torch.tensor([self.rollout_depth]).long())], dim=-1)
                time = time[0]
            else:
                if not self.flex_t:
                    time = F.one_hot(torch.tensor(cur_t).long(), self.rec_t)
                else:
                    time = torch.tensor([self.discounting ** (self.cur_t)])                    
                
            if not self.flex_t:
                ret_list = [root_node_stat, cur_node_stat, reset, time, depc, root_trail_r, root_rollout_q, root_max_q]
            else:
                term = torch.tensor([term], dtype=torch.float32)                            
                ret_list = [root_node_stat, cur_node_stat, root_trail_r, root_rollout_q, root_max_q, reset, depc, term, time]
                
            out = torch.concat(ret_list, dim=-1)  
            self.last_node = self.cur_node     
            
            self.root_max_q = root_max_q
            self.ret_dict = {"v0": self.root_node.ret_dict["v"].unsqueeze(0),
                             "q_s_a": self.root_node.ret_dict["child_rollout_qs_mean"].unsqueeze(0),
                             "max_q_s_a": self.root_node.ret_dict["child_rollout_qs_max"].unsqueeze(0),
                             "n_s_a": self.root_node.child_rollout_ns.unsqueeze(0),
                             "logit0": self.root_node.ret_dict["child_logits"].unsqueeze(0),
                             "logit": self.cur_node.ret_dict["child_logits"].unsqueeze(0),
                             "reset": reset,
                             "term": term}
            
            if self.thres is not None:
                self.ret_dict["thres"] = self.thres
            
            if reset:
                self.rollout_depth = 0
                self.unexpand_rollout_depth = 0.
                self.cur_node = self.root_node
                self.cur_node.visit()
                self.pass_unexpand = False
            
            return out
                
class Node:
    def __init__(self, parent, action, logit, num_actions, discounting, rec_t):        
        
        self.action = F.one_hot(torch.tensor(action).long(), num_actions) # shape (1, num_actions)        
        self.r = torch.tensor([0.], dtype=torch.float32)    
        self.v = torch.tensor([0.], dtype=torch.float32)            
        self.logit = logit # shape (1,)        
        
        self.rollout_qs = []  # list of tensors of shape (1,)
        self.rollout_n = torch.tensor([0.], dtype=torch.float32)    
        self.parent = parent
        self.children = []
        self.encoded = None 
        
        self.num_actions = num_actions
        self.discounting = discounting
        self.rec_t = rec_t        
        
        self.visited = False

    def expanded(self):
        return len(self.children) > 0

    def expand(self, r, v, logits, encoded, override=False):
        """
        First time arriving a node and so we expand it
        r, v: tensor of shape (1,)
        logits: tensor of shape (num_actions,)
        """
        if not override: assert not self.expanded()
        if override:
            self.rollout_qs = [x - self.r + r for x in self.rollout_qs]
            self.rollout_qs[0] = v * self.discounting
        self.r = r
        self.v = v
        self.encoded = encoded
        for a in range(self.num_actions):
            if not override:
                child = self.children.append(Node(self, a, logits[[a]], 
                   self.num_actions, self.discounting, self.rec_t))
            else:
                self.children[a].logit = logits[[a]]        
            
    def visit(self):
        self.trail_r = torch.tensor([0.], dtype=torch.float32)    
        self.trail_discount = 1.
        self.propagate(self.r, self.v, not self.visited)        
        self.visited = True
        
    def propagate(self, r, v, new_rollout):
        self.trail_r = self.trail_r + self.trail_discount * r
        self.trail_discount = self.trail_discount * self.discounting
        self.rollout_q = self.trail_r + self.trail_discount * v
        if new_rollout:
            self.rollout_qs.append(self.rollout_q)
            self.rollout_n = self.rollout_n + 1
        if self.parent is not None: self.parent.propagate(r, v, new_rollout)
            
    def stat(self, pos=None):
        assert self.expanded()
        self.child_logits = torch.concat([x.logit for x in self.children])        
        child_rollout_qs_mean = []
        child_rollout_qs_max = []
        for x in self.children:
            if len(x.rollout_qs) > 0:                
                q_mean = torch.mean(torch.cat(x.rollout_qs), dim=-1, keepdim=True)
                q_max = torch.max(torch.cat(x.rollout_qs), dim=-1, keepdim=True)[0]
            else:
                q_mean = torch.tensor([0.], dtype=torch.float32)    
                q_max = torch.tensor([0.], dtype=torch.float32)    
            child_rollout_qs_mean.append(q_mean)
            child_rollout_qs_max.append(q_max)
        self.child_rollout_qs_mean = torch.concat(child_rollout_qs_mean)
        self.child_rollout_qs_max = torch.concat(child_rollout_qs_max)
        
        self.child_rollout_ns = torch.tensor([x.rollout_n for x in self.children]).long()
        if pos is None:
            self.child_rollout_ns_enc = self.child_rollout_ns / self.rec_t       
        else:
            self.child_rollout_ns_enc = torch.flatten(pos(self.child_rollout_ns))
            
        ret_list = ["action", "r", "v", "child_logits", "child_rollout_qs_mean",
                    "child_rollout_qs_max", "child_rollout_ns_enc"]
        self.ret_dict = {x: getattr(self, x) for x in ret_list}
        out = torch.concat(list(self.ret_dict.values()))        
        return out        
    
def define_parser():

    parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

    parser.add_argument("--env", type=str, default="Sokoban-v0",
                        help="Gym environment.")
    parser.add_argument("--env_disable_noop", action="store_true",
                        help="Disable noop in environment or not. (sokoban only)")

    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")

    parser.add_argument("--disable_checkpoint", action="store_true",
                        help="Disable saving checkpoint.")
    parser.add_argument("--load_checkpoint", default="",
                        help="Load checkpoint directory.")    
    parser.add_argument("--savedir", default="~/RS/thinker/logs/torchbeast",
                        help="Root dir where experiment data will be saved.")

    # Training settings.        
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
    parser.add_argument("--tran_lstm_new", action="store_true",
                        help="Whether to use a speed-up version of LSTM-transformer.")    
    parser.add_argument("--tran_lstm_no_attn", action="store_true",
                        help="Whether to disable attention in LSTM-transformer.")
    parser.add_argument("--tran_attn_b", default=5.,
                        type=float, help="Bias attention for current position.")    
    parser.add_argument("--tran_erasep", action="store_true",
                        help="Whether to erase past memories if not planning.")
    
    
    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.0001,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--im_entropy_cost", default=0.0001,
                        type=float, help="Imagainary Entropy cost/multiplier.")         
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--reg_cost", default=0.1,
                        type=float, help="Reg cost/multiplier.")
    parser.add_argument("--real_cost", default=1,
                        type=float, help="Real reward - real action cost/multiplier.")      
    parser.add_argument("--real_im_cost", default=1,
                        type=float, help="Real reward - imagainary action cost/multiplier.")          
    parser.add_argument("--im_cost", default=1,
                        type=float, help="Imaginary reward cost/multiplier.")   
    parser.add_argument("--discounting", default=0.99,
                        type=float, help="Discounting factor.")
    parser.add_argument("--lamb", default=1.,
                        type=float, help="Lambda when computing trace.")
    parser.add_argument("--reward_clipping", default=10, type=int, 
                        metavar="N", help="Reward clipping.")
    parser.add_argument("--trun_bs", action="store_true",
                        help="Whether to add baseline as reward when truncated.")
    
    # Model settings
    parser.add_argument("--reward_type", default=1, type=int, metavar="N",
                        help="Reward type")   
    parser.add_argument("--reset_m", default=-1, type=int, metavar="N",
                        help="Auto reset after passing m node since an unexpanded noded")    
    parser.add_argument("--model_type_nn", default=0,
                        type=float, help="Model type.")     
    parser.add_argument("--perfect_model", action="store_true",
                        help="Whether to use perfect model.")    
    parser.add_argument("--stat_pos_encode", action="store_true",
                        help="Whether to use positional encoding for integers")       
    parser.add_argument("--stat_pos_encode_dim", default=32, type=int, metavar="N",
                        help="Dimension of positional encoding (only enabled when stat_pos_encode == True).")        
    parser.add_argument("--rec_t", default=5, type=int, metavar="N",
                        help="Number of planning steps.")
    parser.add_argument("--flex_t", action="store_true",
                        help="Whether to enable flexible planning steps.") 
    parser.add_argument("--flex_t_cost", default=-1e-5,
                        type=float, help="Cost of planning step (only enabled when flex_t == True).")
    parser.add_argument("--flex_t_cost_m", default=-1e-2,
                        type=float, help="Multipler to exp. of planning cost (only enabled when flex_t_cost_type == 1).")    
    parser.add_argument("--flex_t_cost_type", default=0,
                        type=int, help="Type of planning cost; 0 for constant, 1 for exp. decay")                    
    parser.add_argument("--flex_t_term_b", default=-5,
                        type=float, help="Bias added to the logit of term action.")      
    parser.add_argument("--no_mem", action="store_true",
                        help="Whether to erase all memories after each real action.")   
    parser.add_argument("--tree_carry", action="store_true",
                        help="Whether to carry over the tree.")   
    parser.add_argument("--tree_vb", default=0., type=float,
                        help="Adjustment to initial max-Q.")    
    parser.add_argument("--thres_carry", action="store_true",
                        help="Whether to carry threshold over.")   
    parser.add_argument("--reward_carry", action="store_true",
                        help="Whether to carry planning reward over.")      
    parser.add_argument("--thres_discounting", default=0.99,
                        type=float, help="Threshold discounting factor.")    
    

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
flags = parser.parse_args([])        

flags.xpid = None
flags.load_checkpoint = ""

flags.env = "cSokoban-v0"
flags.num_actors = 1
flags.batch_size = 8
flags.unroll_length = 50
flags.learning_rate = 0.0001
flags.grad_norm_clipping = 60

flags.entropy_cost = 0.00001
flags.im_entropy_cost = 0.00001
flags.reg_cost = 0.01
flags.real_cost = 1
flags.real_im_cost = 1
flags.im_cost = 1
flags.discounting = 0.97
flags.lamb = 1.

flags.trun_bs = False
flags.total_steps = 500000000
flags.disable_adam = False

flags.tran_t = 1
flags.tran_mem_n = 5
flags.tran_layer_n = 3
flags.tran_lstm = True
flags.tran_lstm_no_attn = False
flags.tran_attn_b = 5
flags.tran_norm_first = False
flags.tran_ff_n = 256
flags.tran_skip = False
flags.tran_erasep = False
flags.tran_dim = 64
flags.tran_rpos = True

flags.no_mem = True
flags.rec_t = 10
flags.model_type_nn = 0
flags.perfect_model = True
flags.reward_type = 1
flags.stat_pos_encode = False
flags.stat_pos_encode_dim = 32

flags.reset_m = -1
flags.tree_carry = True
flags.thres_carry = True
flags.reward_carry = False
flags.thres_discounting = 0.97
flags.flex_t = False
flags.flex_t_cost = 1e-6
flags.flex_t_cost_m = 1e-2
flags.flex_t_cost_type = 0
flags.flex_t_term_b = -3.

flags.savedir = "~/tmp"    