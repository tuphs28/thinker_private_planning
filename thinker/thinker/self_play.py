import time
import numpy as np
import argparse
from collections import namedtuple
import torch
from torch import nn
import ray
from thinker.buffer import ActorBuffer, ParamBuffer
from thinker.buffer import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.net import ActorNet, ModelNet, ActorOut
from thinker.env import Environment, EnvOut
import thinker.util as util

_fields = tuple(item for item in ActorOut._fields + EnvOut._fields if item != 'gym_env_out')
TrainActorOut = namedtuple('TrainActorOut', _fields)

@ray.remote
class SelfPlayWorker():
    def __init__(self, param_buffer:ParamBuffer, actor_buffer: ActorBuffer, rank: int, flags:argparse.Namespace):        
        self.param_buffer = param_buffer
        self.actor_buffer = actor_buffer
        self.rank = rank
        self.flags = flags

        self.env = Environment(flags)
        self.actor_net = ActorNet(obs_shape=self.env.model_out_shape, num_actions=self.env.num_actions, flags=flags)
        self.model_net = ModelNet(obs_shape=self.env.gym_env_out_shape, num_actions=self.env.num_actions, flags=flags)
        self.model_net.train(False)

        if rank == 0:
            if self.flags.preload_model:
                checkpoint = torch.load(self.flags.preload_model, map_location=torch.device('cpu'))
                self.model_net.set_weights(checkpoint["model_state_dict"])  
                print("Loadded model network from %s" % self.flags.preload_model)
            self.param_buffer.set_weights.remote("model_net", self.model_net.get_weights())
        
        # synchronize weight before start
        while(True):
            weights = ray.get(self.param_buffer.get_weights.remote("actor_net")) # set by actor_learner
            if weights is not None:
                self.actor_net.set_weights(weights)
                break
            time.sleep(0.1)

        while(True):
            weights = ray.get(self.param_buffer.get_weights.remote("model_net")) # set by rank 0 self_play_worker
            if weights is not None:
                self.model_net.set_weights(weights)
                break
            time.sleep(0.1)    
                

    def write_buffer(self, env_out: EnvOut, actor_out: ActorOut, t: int):
        if t == 0:
            fields = {}
            for field in TrainActorOut._fields:
                out = getattr(env_out if field in EnvOut._fields else actor_out, field)
                if out is not None:
                    fields[field] = torch.empty(size=(self.flags.unroll_length+1, 1)+out.shape[2:], dtype=out.dtype)
                    # each is in the shape of (T x B xdim_1 x dim_2 ...)
                else:
                    fields[field] = None
            self.buffer = TrainActorOut(**fields)
            
        for field in TrainActorOut._fields:
            v = getattr(self.buffer, field)            
            if v is not None:
                v[t] = getattr(env_out if field in EnvOut._fields else actor_out, field)[0,0]
        
        if t == self.flags.unroll_length:
            # post-processing
            self.buffer = util.tuple_map(self.buffer, lambda x: x.cpu())

    def gen_data(self):
        with torch.no_grad():
            print("Actor %d started." % self.rank)
            n = 0            

            env_out = self.env.initial(self.model_net)            
            actor_state = self.actor_net.initial_state(batch_size=1)
            actor_out, _ = self.actor_net(env_out, actor_state)
            while (True):      
                # prepare train_actor_out data to be written
                initial_actor_state = actor_state
                self.write_buffer(env_out, actor_out, 0)
                for t in range(self.flags.unroll_length):
                    actor_out, actor_state = self.actor_net(env_out, actor_state)    
                    action = [actor_out.action, actor_out.im_action, actor_out.reset_action]
                    if actor_out.term_action is not None:
                        action.append(actor_out.term_action)
                    action = torch.cat(action, dim=-1)
                    env_out = self.env.step(action.unsqueeze(0), self.model_net)
                    self.write_buffer(env_out, actor_out, t+1)                
                train_actor_out = (self.buffer, initial_actor_state)

                # send the data to remote
                status = 0
                while (True):
                    status = ray.get(self.actor_buffer.get_status.remote())
                    if status == AB_FULL: time.sleep(0.01) 
                    else: break
                if status == AB_FINISH: break
                train_actor_out = ray.put(train_actor_out)
                self.actor_buffer.write.remote([train_actor_out])

                # set model weight                
                if n % 1 == 0:
                    weights = ray.get(self.param_buffer.get_weights.remote("actor_net"))
                    self.actor_net.set_weights(weights)
                    weights = ray.get(self.param_buffer.get_weights.remote("model_net"))
                    self.model_net.set_weights(weights)                
                n += 1

        return True