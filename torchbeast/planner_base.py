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
from torchbeast.base import BaseNet

import gym
import gym_sokoban
import numpy as np
from matplotlib import pyplot as plt
import logging
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

def gumbel_softmax(logits, temperature, u=None, hard=True):
    """
    ST-gumple-softmax
    input: [bsz, n_class]
    return: flatten --> [bsz, n_class] an one-hot vector
    """
    eps = 1e-20
    n_class = logits.shape[-1]
    
    if u is None:
        u = torch.rand(logits.size()).to(logits.device)
        u = -torch.log(-torch.log(u + eps) + eps)
    y = logits + u
    y = F.softmax(y / temperature, dim=-1)
    
    if not hard: return y.view(-1, n_class)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, n_class), u

class Actor_Wrapper(nn.Module):    
    def __init__(self, flags, model, actor=None):
        
        super(Actor_Wrapper, self).__init__()   
        self.model = model
        self.num_actions = model.num_actions        
        self.rec_t = flags.rec_t
        self.discounting = flags.discounting
        self.aug_stat = flags.aug_stat
        
        obs_n = (7 + num_actions * 7 + self.rec_t if self.aug_stat else 
            5 + num_actions * 3 + self.rec_t)       
        if actor is None:
            self.actor = Actor_net(obs_shape=(obs_n, 1, 1), num_actions=self.num_actions, flags=flags)
        else:
            self.actor = actor   
        self.use_model = self.use_model_aug if self.aug_stat else self.use_model_base        
            
    def initial_state(self, batch_size):
        return self.actor.initial_state(batch_size)
            
    def forward(self, x, core_state=None):                      
        # x is env_output object with:
        # frame: T x B x C x H x W
        # last_action: T x B
        # reward: T x B
        
        tot_step, bsz, _, _, _ = x['frame'].shape
        device = x['frame'].device        
        self.model.train(False)
        
        for step in range(tot_step):
        
            state = x['frame'][step]        
            action = F.one_hot(x['last_action'][step], self.num_actions)
            reward = x['reward'][step]   
            done = x['done'][step]
            reset = torch.ones(bsz, device=device)
            
            u_list, im_logit_list, reset_logit_list = [], [], []                
            for t in range(self.rec_t):                 
                actor_input = self.use_model(state, reward, action, t, reset)                
                reset_ex = reset.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                self.encoded = reset_ex * self.encoded_reset + (1 - reset_ex) * self.encoded               
                
                if 'uniform' in x.keys():
                    u = x['uniform'][step][:, t]
                else:
                    u = None                    
                actor_output, core_state = self.actor(actor_input, 
                                                      done=done, 
                                                      core_state=core_state,
                                                      u=u)                
                if self.actor.gb_ste: u_list.append(actor_output['uniform'].unsqueeze(1))
                im_logit_list.append(actor_output['im_policy_logits'].unsqueeze(1))
                reset_logit_list.append(actor_output['reset_policy_logits'].unsqueeze(1))
                
                action = actor_output["im_action"]
                reset = actor_output["reset"]
        
            if self.actor.gb_ste:
                actor_output["uniform"] = torch.concat(u_list, dim=1)
                actor_output['im_policy_logits'] = torch.concat(im_logit_list, dim=1)
                actor_output['reset_policy_logits'] = torch.concat(reset_logit_list, dim=1)
            if step == 0:
                all_actor_output = {k: [v.unsqueeze(0)] for k, v in actor_output.items()}
            else:
                for k, v in actor_output.items(): all_actor_output[k].append(v.unsqueeze(0))        
        
        all_actor_output = {k: torch.concat(v) for k, v in all_actor_output.items()}
        return all_actor_output, core_state      
    
    def use_model_aug(self, x, r, a, cur_t, reset):
        # input: 
        # r: reward - [B]; x: frame - [B, C, H, W]; a: action - [B, num_actions]
        # cur_t: int; reset at cur_t == 0  
        device = a.device
        bsz = a.shape[0]
        if cur_t == 0:
            self.rollout_depth = torch.zeros(bsz, dtype=torch.float32, device=device)
            self.re_action = a
            _, vs, logits, encodeds = self.model(x, self.re_action.unsqueeze(0), one_hot=True)                
            self.encoded = encodeds[-1]    
            self.encoded_reset = self.encoded.clone()

            self.re_reward = r.unsqueeze(-1)              
            self.v0 = vs[-1].unsqueeze(-1).clone()
            self.logit0 = logits[-1].clone()

            self.im_action = torch.zeros(bsz, self.num_actions, dtype=torch.float32, device=device)
            self.im_reset = torch.ones(bsz, 1, dtype=torch.float32, device=device)
            self.im_reward = torch.zeros(bsz, 1, dtype=torch.float32, device=device)                              
            self.v = vs[-1].unsqueeze(-1)
            self.logit = logits[-1]
            self.rollout_first_action = torch.zeros(bsz, self.num_actions, dtype=torch.float32, device=device)  
            self.rollout_return_wo_v = torch.zeros(bsz, 1, dtype=torch.float32, device=device)     
            self.rollout_return = torch.zeros(bsz, 1, dtype=torch.float32, device=device)     
            self.q_s_a = torch.zeros(bsz, self.num_actions, dtype=torch.float32, device=device)      
            self.n_s_a = torch.zeros(bsz, self.num_actions, dtype=torch.float32, device=device)                   
        else:
            self.rollout_depth = self.rollout_depth + 1                

            self.im_action = a
            rs, vs, logits, encodeds = self.model.forward_encoded(self.encoded, 
               self.im_action.unsqueeze(0), one_hot=True)
            self.encoded = encodeds[-1]        

            self.im_reward = rs[-1].unsqueeze(-1)
            self.v = vs[-1].unsqueeze(-1)    
            self.logit = logits[-1]     

            self.rollout_first_action = (self.im_reset * self.im_action + (1 - self.im_reset) * 
                self.rollout_first_action)
            self.rollout_return_wo_v = (self.im_reset * torch.zeros_like(self.rollout_return_wo_v) + 
                                        (1 - self.im_reset) * self.rollout_return_wo_v)
            self.rollout_depth = (self.im_reset[:,0] * torch.ones_like(self.rollout_depth) + 
                                        (1 - self.im_reset)[:,0] * self.rollout_depth)

            self.rollout_return_wo_v = (self.rollout_return_wo_v + (self.discounting ** (self.rollout_depth-1)
                                                                   ).unsqueeze(-1) * self.im_reward)
            self.rollout_return = self.rollout_return_wo_v + (
                self.discounting ** (self.rollout_depth).unsqueeze(-1)) * self.v                    
            
            self.im_reset = reset.unsqueeze(-1)
            
            new_q_s_a = self.q_s_a * self.n_s_a / (self.n_s_a + 1) + self.rollout_return / (self.n_s_a + 1)
            new_q_s_a = new_q_s_a * self.rollout_first_action + self.q_s_a * (1 - self.rollout_first_action)
            self.q_s_a = (self.im_reset * new_q_s_a + (1 - self.im_reset) * self.q_s_a)
            self.n_s_a = (self.im_reset * (self.n_s_a + self.rollout_first_action) + 
                (1 - self.im_reset) * self.n_s_a)
            
            
        time = F.one_hot(torch.tensor([cur_t], device=device).long(), self.rec_t).tile([bsz, 1])
        depc = self.discounting ** (self.rollout_depth-1).unsqueeze(-1)
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
        out = out.unsqueeze(-1).unsqueeze(-1)  
        return out          
    
    def use_model_base(self, x, r, a, cur_t, reset):
        bsz = a.shape[0]
        if cur_t == 0:
            rs, vs, logits, encodeds = self.model(x, a.unsqueeze(0), one_hot=True)
            self.encoded = encodeds[0] 
            self.encoded_reset = encodeds[0].clone()
            self.r0 = r.unsqueeze(-1).clone()
            self.v0 = vs[-1].unsqueeze(-1).clone()
            self.logit0 = logits[-1].clone() 
            r = r.unsqueeze(-1)
            v = vs[-1].unsqueeze(-1)
            logit = logits[-1]     
        else:
            rs, vs, logits, encodeds = self.model.forward_encoded(self.encoded, 
                a.unsqueeze(0), one_hot=True)
            self.encoded = encodeds[-1] 
            r = rs[-1].unsqueeze(-1)
            v = vs[-1].unsqueeze(-1)
            logit = logits[-1]  
            
        re = reset.unsqueeze(-1)
        time = F.one_hot(torch.tensor([cur_t], device=a.device).long(), self.rec_t).tile([bsz, 1])                        

        actor_input = torch.concat([re, a, r, v, logit, self.r0, self.v0, self.logit0, time], dim=-1)     
        actor_input = actor_input.unsqueeze(-1).unsqueeze(-1)    
        return actor_input

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
        self.ste = flags.ste
        self.gb_ste = flags.gb_ste
        
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
                                       num_layers=self.tran_layer_n, rpos=self.rpos, conv=False)   
                         
        
        if self.tran_skip:
            rnn_out_size = self.conv_out_hw * self.conv_out_hw * (self.d_model + self.env_input_size)
        else:
            rnn_out_size = self.conv_out_hw * self.conv_out_hw * self.d_model
                
        self.fc = nn.Linear(rnn_out_size, 256)        
        
        self.im_policy = nn.Linear(256, self.num_actions)        
        self.policy = nn.Linear(256, self.num_actions)        
        self.baseline = nn.Linear(256, 1)        
        self.reset = nn.Linear(256, 1)        
        
        if self.gb_ste:
            self.register_buffer('temp', torch.tensor(flags.gb_ste_temp_max, dtype=torch.float32))
        
        print("actor size: ", sum(p.numel() for p in self.parameters()))
        #for k, v in self.named_parameters(): print(k, v.numel())   

    def initial_state(self, batch_size):
        state = self.core.init_state(batch_size) + (torch.zeros(1, batch_size, 
               self.env_input_size, self.conv_out_hw, self.conv_out_hw),)
        return state

    def forward(self, x, done, core_state=(), u=None, debug=False):
        # one-step forward for the actor
        # input / done shape x: T x B x C x 1 x 1 / B x C x 1 x 1
        # only supports T = 1 at the moment; all output does not have T dim.
        
        if len(x.shape) == 4: x = x.unsqueeze(0)
        if len(done.shape) == 1: done = done.unsqueeze(0)  
            
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.        
        env_input = self.frame_conv(x)                
        core_input = env_input.view(T, B, -1, self.conv_out_hw, self.conv_out_hw)
        core_output_list = []
        notdone = ~(done.bool())
        
        for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):                
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
        reset_policy_logits_p = self.reset(core_output)
        reset_policy_logits = torch.cat([reset_policy_logits_p, torch.zeros_like(reset_policy_logits_p)], dim=-1)   
        baseline = self.baseline(core_output)
        
        if self.gb_ste:
            if u is not None: 
                u_action = u[:, :-2]
                u_reset = u[:, -2:]
            else:
                u_action = None
                u_reset = None
            im_action, u_action = gumbel_softmax(im_policy_logits, self.temp, u=u_action, hard=True)             
            reset, u_reset = gumbel_softmax(reset_policy_logits, self.temp, u=u_reset, hard=True)
            reset = reset[:, 0]
            u = torch.cat([u_action, u_reset], dim=-1)
        elif self.ste:
            im_action_p = F.softmax(im_policy_logits, dim=-1)
            im_action_h = torch.multinomial(im_action_p, num_samples=1)
            im_action = (im_action_h - im_action_p).detach() + im_action_p
          
            reset_p = torch.sigmoid(reset_policy_logits_p)
            reset_h = torch.bernoulli(reset_p)
            reset = (reset_h - reset_p).detach() + reset_p
        else:
            im_action = F.softmax(im_policy_logits, dim=1)
            reset = torch.sigmoid(reset_policy_logits_p)
            
        
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        
        reg_loss = (1e-3 * torch.sum(policy_logits**2, dim=-1) / 2 + 
                    1e-5 * torch.sum(core_output**2, dim=-1) / 2)
        reg_loss = reg_loss.view(T, B)
        
        policy_logits = policy_logits.view(T, B, self.num_actions)
        action = action.view(T, B)                
        im_policy_logits = im_policy_logits.view(T, B, self.num_actions)        
        im_action = im_action.view(T, B, self.num_actions)                
        reset_policy_logits = reset_policy_logits.view(T, B, 2)
        reset = reset.view(T, B)     
        baseline = baseline.view(T, B)
        
        ret_dict = dict(policy_logits=policy_logits[0],                         
                        action=action[0], 
                        im_policy_logits=im_policy_logits[0],                         
                        im_action=im_action[0],                                                
                        reset_policy_logits=reset_policy_logits[0], 
                        reset=reset[0],
                        baseline=baseline[0], 
                        reg_loss=reg_loss[0], )
        if self.gb_ste:
            ret_dict['uniform'] = u
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
    parser.add_argument("--total_steps", default=100000000, type=int, metavar="T",
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

    # Architecture settings
    parser.add_argument("--tran_dim", default=64, type=int, metavar="N",
                        help="Size of transformer hidden dim.")
    parser.add_argument("--tran_mem_n", default=16, type=int, metavar="N",
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
    
    parser.add_argument("--ste", action="store_true",
                        help="Whether to use ste backprop.")
    parser.add_argument("--gb_ste", action="store_true",
                        help="Whether to use gb-ste backprop.")
    parser.add_argument("--gb_ste_temp_max", default=1, type=int, metavar="N",
                        help="Beginning temp. for gb-ste.")
    parser.add_argument("--gb_ste_temp_min", default=0.5, type=int, metavar="N",
                        help="Ending temp. for gb-ste.")    

    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.01,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--im_entropy_cost", default=0.01,
                        type=float, help="Entropy cost/multiplier.")    
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
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
flags = parser.parse_args()        

env = create_env(flags)
obs_shape, num_actions = env.observation_space.shape, env.action_space.n
model_learner = Model(flags, obs_shape, num_actions=num_actions)
model_actor = Model(flags, obs_shape, num_actions=num_actions)
checkpoint = torch.load("../models/model_1.tar")
model_learner.load_state_dict(checkpoint["model_state_dict"])  
model_actor.load_state_dict(checkpoint["model_state_dict"])  

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

env = create_env(flags)

actor_net = Actor_Wrapper(flags, model_actor, actor=None)
buffers = create_buffers(flags, env.observation_space.shape, env.action_space.n)

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
            actor_net, buffers, initial_agent_state_buffers,),)
    actor.start()
    actor_processes.append(actor)

learner_net = Actor_Wrapper(flags, model_learner, actor=None, 
                            ).to(device=flags.device)

if not flags.disable_adam:
    print("Using Adam...")        
    optimizer = torch.optim.Adam(learner_net.actor.parameters(),lr=flags.learning_rate)
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
        if flags.gb_ste:
            learner_net.actor.temp = torch.tensor(np.maximum(flags.gb_ste_temp_max * 
                    np.exp(-step/flags.total_steps), flags.gb_ste_temp_min).item(), 
                    dtype=torch.float32, device=learner_net.actor.temp.device)
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

        for s in ["pg_loss", "baseline_loss", "entropy_loss", "im_entropy_loss", "reg_loss"]:
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