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

from torchbeast.core.environment import Environment, Vec_Environment
from torchbeast.atari_wrappers import EnvWrapper
from torchbeast.base import BaseNet
from torchbeast.train import create_env

import gym
import gym_sokoban
import numpy as np
import logging
from matplotlib import pyplot as plt
from collections import deque

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

torch.multiprocessing.set_sharing_strategy('file_system')

def get_param(net, name=None):
    keys = []
    for (k, v) in actor_wrapper.model.named_parameters(): 
        if name is None:
            print(k)
        else:
            if name == k: return v
        keys.append(k)
    return keys        

def n_step_greedy(env, net, n):    
    if isinstance(env, Vec_Environment):
        num_actions = env.gym_env.action_space[0].n
        bsz = len(env.gym_env.envs)
    else:
        num_actions = env.gym_env.action_space.n
        bsz = 1

    temp = 10.
    q_ret = torch.zeros(bsz, num_actions).to(device)      
    state = env.clone_state()

    for act in range(num_actions):
        obs = env.step(torch.Tensor(np.full(bsz, act)).long())      
        obs = {k:v.to(device) for k, v in obs.items()}   
        
        if n > 1:
            action, prob, sub_q_ret = n_step_greedy(env, net, n-1)
            ret = obs['reward'] + flags.discounting * torch.max(sub_q_ret, dim=1)[0] * (~obs['done']).float()
        else:
            ret = obs['reward'] + flags.discounting * net(obs)[0]['baseline'] * (~obs['done']).float()

        q_ret[:, act] = ret
        env.restore_state(state)
    
    prob = F.softmax(temp*q_ret, dim=1)
    action = torch.multinomial(prob, num_samples=1)[:, 0]
    
    return action, prob, q_ret  
# Generating data for learning model [RUN]

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_buffers_m(flags, obs_shape, num_actions) -> Buffers:
    
    seq_len = flags.seq_len
    seq_n = flags.seq_n
    specs = dict(
        frame=dict(size=(seq_len + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(seq_len + 1,), dtype=torch.float32),
        done=dict(size=(seq_len + 1,), dtype=torch.bool),
        truncated_done=dict(size=(seq_len + 1,), dtype=torch.bool),
        episode_return=dict(size=(seq_len + 1,), dtype=torch.float32),
        episode_step=dict(size=(seq_len + 1,), dtype=torch.int32),
        policy_logits=dict(size=(seq_len + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(seq_len + 1,), dtype=torch.float32),
        last_action=dict(size=(seq_len + 1,), dtype=torch.int64),
        action=dict(size=(seq_len + 1,), dtype=torch.int64),
        reg_loss=dict(size=(seq_len + 1,), dtype=torch.float32)
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(seq_n):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
            
    return buffers

def gen_data(
    flags,
    actor_index: int,
    net: torch.nn.Module,
    buffers: Buffers,
    free_queue: mp.SimpleQueue,
):    
    try:    
        #logging.info("Actor %i started", actor_index)
        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = Environment(gym_env)
        env_output = env.initial()  
        agent_state = net.initial_state(batch_size=1)
        agent_output, unused_state = net(env_output, agent_state)     
        
        while True:
            index = free_queue.get()
            if index is None:
                break         

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]

            # Do new rollout.
            for t in range(flags.seq_len):
                with torch.no_grad():
                    agent_output, agent_state = net(env_output, agent_state)
                env_output = env.step(agent_output["action"])
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                    
    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        raise e
        

# Models

DOWNSCALE_C = 2

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation,)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, outplanes=None):
        super().__init__()
        if outplanes is None: outplanes = inplanes 
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        self.skip_conv = (outplanes != inplanes)
        if outplanes != inplanes:
            self.conv3 = conv1x1(inplanes, outplanes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip_conv:
            out += self.conv3(identity)
        else:
            out += identity
        out = self.relu(out)
        return out
    
class FrameEncoder(nn.Module):    
    def __init__(self, num_actions, frame_channels=3, type_nn=0):
        self.num_actions = num_actions
        super(FrameEncoder, self).__init__() 
        
        if type_nn == 0:
            n_block = 1
        elif type_nn == 1:
            n_block = 2
        
        self.conv1 = nn.Conv2d(in_channels=frame_channels+num_actions, out_channels=128//DOWNSCALE_C, kernel_size=3, stride=2, padding=1) 
        res = nn.ModuleList([ResBlock(inplanes=128//DOWNSCALE_C) for i in range(n_block)]) # Deep: 2 blocks here
        self.res1 = torch.nn.Sequential(*res)
        self.conv2 = nn.Conv2d(in_channels=128//DOWNSCALE_C, out_channels=256//DOWNSCALE_C, 
                               kernel_size=3, stride=2, padding=1) 
        res = nn.ModuleList([ResBlock(inplanes=256//DOWNSCALE_C) for i in range(n_block)]) # Deep: 3 blocks here
        self.res2 = torch.nn.Sequential(*res)
        self.avg1 = nn.AvgPool2d(3, stride=2, padding=1)
        res = nn.ModuleList([ResBlock(inplanes=256//DOWNSCALE_C) for i in range(n_block)]) # Deep: 3 blocks here
        self.res3 = torch.nn.Sequential(*res)
        self.avg2 = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(self, x, actions):        
        # input shape: B, C, H, W        
        # action shape: B 
        
        x = x.float() / 255.0    
        actions = actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])        
        x = torch.concat([x, actions], dim=1)
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = self.avg1(x)
        x = self.res3(x)
        x = self.avg2(x)
        return x
    
class DynamicModel(nn.Module):
    def __init__(self, num_actions, inplanes=256, type_nn=0):        
        super(DynamicModel, self).__init__()
        self.type_nn = type_nn
        if type_nn == 0:
            res = nn.ModuleList([ResBlock(inplanes=inplanes+num_actions, outplanes=inplanes)] + [
                    ResBlock(inplanes=inplanes) for i in range(4)]) 
        elif type_nn == 1:                      
            res = nn.ModuleList([ResBlock(inplanes=inplanes) for i in range(15)] + [
                    ResBlock(inplanes=inplanes, outplanes=inplanes*num_actions)])

        
        self.res = torch.nn.Sequential(*res)
        self.num_actions = num_actions
    
    def forward(self, x, actions):              
        bsz, c, h, w = x.shape
        if self.training:
            x.register_hook(lambda grad: grad * 0.5)
        if self.type_nn == 0:
            actions = actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])        
            x = torch.concat([x, actions], dim=1)
            out = self.res(x)
        elif self.type_nn == 1:            
            res_out = self.res(x).view(bsz, self.num_actions, c, h, w)        
            actions = actions.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = torch.sum(actions * res_out, dim=1)
        return out
    
class Output_rvpi(nn.Module):   
    def __init__(self, num_actions, input_shape):         
        super(Output_rvpi, self).__init__()        
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=3, padding='same') 
        self.conv2 = nn.Conv2d(in_channels=c//2, out_channels=c//4, kernel_size=3, padding='same') 
        fc_in = h * w * (c // 4)
        self.fc_r = nn.Linear(fc_in, 1) 
        self.fc_v = nn.Linear(fc_in, 1) 
        self.fc_logits = nn.Linear(fc_in, num_actions)         
        
    def forward(self, x):    
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        r, v, logits = self.fc_r(x), self.fc_v(x), self.fc_logits(x)
        return r, v, logits

class Model(nn.Module):    
    def __init__(self, flags, obs_shape, num_actions):        
        super(Model, self).__init__()      
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions          
        self.type_nn = flags.model_type_nn # type_nn: type of neural network for the model; 0 for small, 1 for large
        self.frameEncoder = FrameEncoder(num_actions=num_actions, frame_channels=obs_shape[0], type_nn=self.type_nn)
        self.dynamicModel = DynamicModel(num_actions=num_actions, inplanes=256//DOWNSCALE_C, type_nn=self.type_nn)
        self.output_rvpi = Output_rvpi(num_actions=num_actions, input_shape=(256//DOWNSCALE_C, 
                      obs_shape[1]//16, obs_shape[1]//16))
        
    def forward(self, x, actions, one_hot=False):
        # Input
        # x: frames with shape (B, C, H, W), in the form of s_t
        # actions: action (int64) with shape (k+1, B), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
        # Output
        # reward: predicted reward with shape (k, B), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
        # value: predicted value with shape (k+1, B), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
        # policy: predicted policy with shape (k+1, B), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
        # encoded: encoded states with shape (k+1, B), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
        # Recall the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...
        
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)                
        encoded = self.frameEncoder(x, actions[0])
        return self.forward_encoded(encoded, actions[1:], one_hot=True)
    
    def forward_encoded(self, encoded, actions, one_hot=False):
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)                
        
        r, v, logits = self.output_rvpi(encoded)
        r_list, v_list, logits_list = [], [v.squeeze(-1).unsqueeze(0)], [logits.unsqueeze(0)]
        encoded_list = [encoded.unsqueeze(0)]
        
        for k in range(actions.shape[0]):            
            encoded = self.dynamicModel(encoded, actions[k])
            r, v, logits = self.output_rvpi(encoded)
            r_list.append(r.squeeze(-1).unsqueeze(0))
            v_list.append(v.squeeze(-1).unsqueeze(0))
            logits_list.append(logits.unsqueeze(0))
            encoded_list.append(encoded.unsqueeze(0))        
        
        if len(r_list) > 0:
            rs = torch.concat(r_list, dim=0)
        else:
            rs = None
            
        vs = torch.concat(v_list, dim=0)
        logits = torch.concat(logits_list, dim=0)
        encodeds = torch.concat(encoded_list, dim=0)        
        
        return rs, vs, logits, encodeds

#model = Model(flags, (3, 80, 80), num_actions=5)
#rs, vs, logits = model(torch.rand(16, 3, 80, 80), torch.ones(8, 16).long())

# functions for training models

def get_batch_m(flags, buffers: Buffers):
    batch_indices = np.random.randint(flags.seq_n, size=flags.bsz)
    time_indices = np.random.randint(flags.seq_len - flags.unroll_len, size=flags.bsz)
    batch = {key: torch.stack([buffers[key][m][time_indices[n]:time_indices[n]+flags.unroll_len+1] 
                          for n, m in enumerate(batch_indices)], dim=1) for key in buffers}
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    return batch

def compute_cross_entropy_loss(logits, target_logits, mask):
    target_policy = F.softmax(target_logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return -torch.sum(target_policy * log_policy * (~mask).float().unsqueeze(-1))

def compute_loss_m(model, batch):
    rs, vs, logits, _ = model(batch['frame'][0], batch['action'])
    logits = logits[:-1]

    target_rewards = batch['reward'][1:]
    target_logits = batch['policy_logits'][1:]

    target_vs = []
    target_v = model(batch['frame'][-1], batch['action'][[-1]])[1][0].detach()
    
    for t in range(vs.shape[0]-1, 0, -1):
        new_target_v = batch['reward'][t] + flags.discounting * (target_v * (~batch['done'][t]).float())# +
                           #vs[t-1] * (batch['truncated_done'][t]).float())
        target_vs.append(new_target_v.unsqueeze(0))
        target_v = new_target_v
    target_vs.reverse()
    target_vs = torch.concat(target_vs, dim=0)

    # if done on step j, r_{j}, v_{j-1}, a_{j-1} has the last valid loss 
    # rs is stored in the form of r_{t+1}, ..., r_{t+k}
    # vs is stored in the form of v_{t}, ..., v_{t+k-1}
    # logits is stored in the form of a{t}, ..., a_{t+k-1}

    done_masks = []
    done = torch.zeros(vs.shape[1]).bool().to(batch['done'].device)
    for t in range(vs.shape[0]):
        if t > 0: done = torch.logical_or(done, batch['done'][t])
        done_masks.append(done.unsqueeze(0))

    done_masks = torch.concat(done_masks[:-1], dim=0)
    
    # compute final loss
    huberloss = torch.nn.HuberLoss(reduction='none', delta=1.0)    
    #rs_loss = torch.sum(huberloss(rs, target_rewards.detach()) * (~done_masks).float())
    rs_loss = torch.sum(((rs - target_rewards) ** 2) * (~done_masks).float())
    #vs_loss = torch.sum(huberloss(vs[:-1], target_vs.detach()) * (~done_masks).float())
    vs_loss = torch.sum(((vs[:-1] - target_vs) ** 2) * (~done_masks).float())
    logits_loss = compute_cross_entropy_loss(logits, target_logits.detach(), done_masks)
    
    return rs_loss, vs_loss, logits_loss

# n_step_greedy for testing

def n_step_greedy_model(state, action, model, n, encoded=None, temp=20.): 
    
    # Either input state, action (S_t, A_{t-1}) or the encoded Z_t
    # state / encoded in the shape of (B, C, H, W)
    # action in the shape of (B)    
    with torch.no_grad():    
      bsz = state.shape[0] if encoded is None else encoded.shape[0]
      device = state.device if encoded is None else encoded.device
      num_actions = model.num_actions    

      q_ret = torch.zeros(bsz, num_actions).to(device)        

      for act in range(num_actions):        
          new_action = torch.Tensor(np.full(bsz, act)).long().to(device)    
          if encoded is None:            
              old_new_actions = torch.concat([action.unsqueeze(0), new_action.unsqueeze(0)], dim=0)
              rs, vs, logits, encodeds = model(state, old_new_actions)
          else:
              rs, vs, logits, encodeds = model.forward_encoded(encoded, new_action.unsqueeze(0))

          if n > 1:
              action, prob, sub_q_ret = n_step_greedy_model(state=None, action=None, 
                         model=model, n=n-1, encoded=encodeds[1])
              ret = rs[0] + flags.discounting * torch.max(sub_q_ret, dim=1)[0] 
          else:
              ret = rs[0] + flags.discounting * vs[1]
          q_ret[:, act] = ret

      prob = F.softmax(temp*q_ret, dim=1)
      action = torch.multinomial(prob, num_samples=1)[:, 0]
    
    return action, prob, q_ret        
   
#n_step_greedy_model(batch['frame'][0], batch['action'][0], model, 4)  

def test_n_step_model(n, model, flags, eps_n=100, temp=20.):    
    
    print("Testing %d step planning" % n) 
    
    bsz = 100
    env = gym.vector.SyncVectorEnv([lambda: EnvWrapper(gym.make("Sokoban-v0"), noop=True, name="Sokoban-v0")] * bsz)
    env = Vec_Environment(env, bsz)
    num_actions = env.gym_env.action_space[0].n
    
    model.train(False)
    returns = []
    
    obs = env.initial()
    action = torch.zeros(bsz).long().to(flags.device)
    eps_n_cur = 5

    while(len(returns) <= eps_n):
        cur_returns = obs['episode_return']    
        obs = {k:v.to(flags.device) for k, v in obs.items()}
        new_action, _, _ = n_step_greedy_model(obs['frame'][0], action, model, n, None, temp)        
        obs = env.step(new_action)
        action = new_action
        if torch.any(obs['done']):
            returns.extend(cur_returns[obs['done']].numpy())
        if eps_n_cur <= len(returns) and len(returns) > 0: 
            eps_n_cur = len(returns) + 10
            #print("Finish %d episode: avg. return: %.2f (+-%.2f) " % (len(returns),
            #    np.average(returns), np.std(returns) / np.sqrt(len(returns))))
            
    returns = returns[:eps_n]
    print("Finish %d episode: avg. return: %.2f (+-%.2f) " % (len(returns),
                np.average(returns), np.std(returns) / np.sqrt(len(returns))))
    return returns