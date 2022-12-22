import time
import numpy as np
import argparse
from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
import ray
from thinker.buffer import ActorBuffer, ModelBuffer, GeneralBuffer
from thinker.buffer import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.net import ActorNet, ModelNet, ActorOut
from thinker.env import Environment, EnvOut
import thinker.util as util

_fields = tuple(item for item in ActorOut._fields + EnvOut._fields if item != 'gym_env_out')
TrainActorOut = namedtuple('TrainActorOut', _fields)
TrainModelOut = namedtuple('TrainModelOut', ['gym_env_out', 'policy_logits', 'action', 'reward', 'done'])

PO_NET, PO_MODEL = 0, 1

@ray.remote
class SelfPlayWorker():
    def __init__(self, param_buffer:GeneralBuffer, actor_buffer: ActorBuffer, 
            model_buffer:ModelBuffer, test_buffer:GeneralBuffer,
            policy:int, rank: int, flags:argparse.Namespace):                

        self.param_buffer = param_buffer
        self.actor_buffer = actor_buffer
        self.model_buffer = model_buffer
        self.test_buffer = test_buffer  
        self.policy = policy
        self.rank = rank
        self.flags = flags

        self.env = Environment(flags, model_wrap=policy==PO_NET)
        if self.policy==PO_NET:
            self.actor_net = ActorNet(obs_shape=self.env.model_out_shape, num_actions=self.env.num_actions, flags=flags)

        self.model_net = ModelNet(obs_shape=self.env.gym_env_out_shape, num_actions=self.env.num_actions, flags=flags)
        self.model_net.train(False)

        if flags.train_model: 
            self.model_local_buffer = [self.empty_model_buffer(), self.empty_model_buffer()]        
            self.model_n = 0
            self.model_t = 0

        # the networks weight are set by the respective learner; but if the respective
        # learner does not exist, then rank 0 worker will set the weights
        if rank == 0 and not self.flags.train_actor and self.policy==PO_NET:
            if self.flags.preload_actor:
                checkpoint = torch.load(self.flags.preload_actor, map_location=torch.device('cpu'))
                self.actor_net.set_weights(checkpoint["actor_net_state_dict"])  
                print("Loadded actor network from %s" % self.flags.preload_actor)
            self.param_buffer.set_data.remote("actor_net", self.actor_net.get_weights())

        if rank == 0 and not self.flags.train_model:
            if self.flags.preload_model:
                checkpoint = torch.load(self.flags.preload_model, map_location=torch.device('cpu'))
                self.model_net.set_weights(checkpoint["model_state_dict"] if "model_state_dict" in 
                    checkpoint else checkpoint["model_net_state_dict"])  
                print("Loadded model network from %s" % self.flags.preload_model)
            self.param_buffer.set_data.remote("model_net", self.model_net.get_weights())
        
        # synchronize weight before start
        if self.policy==PO_NET:
            while(True):
                weights = ray.get(self.param_buffer.get_data.remote("actor_net")) # set by actor_learner
                if weights is not None:
                    self.actor_net.set_weights(weights)
                    break
                time.sleep(0.1)

        while(True):
            weights = ray.get(self.param_buffer.get_data.remote("model_net")) # set by rank 0 self_play_worker or model learner
            if weights is not None:
                self.model_net.set_weights(weights)
                break
            time.sleep(0.1)      

        # override model weight if employ_model is True
        if flags.employ_model:            
            self.employ_model_net = ModelNet(obs_shape=self.env.gym_env_out_shape, 
                num_actions=self.env.num_actions, flags=flags, rnn=self.flags.employ_model_rnn)
            checkpoint = torch.load(self.flags.employ_model, map_location=torch.device('cpu'))
            self.employ_model_net.set_weights(checkpoint["model_state_dict"] if "model_state_dict" in 
                    checkpoint else checkpoint["model_net_state_dict"])            
            if rank == 0:  
                print("Override model network from %s" % self.flags.employ_model)
        else:
            self.employ_model_net = self.model_net

    def gen_data(self, test_eps_n:int=0, verbose:bool=True):
        """ Generate self-play data
        Args:
            test_eps_n (int): number of episode to test for (only for testing mode);
            if set to non-zero, the worker will stop once reaching test_eps_n episodes
            and the data will not be sent out to model or actor buffer
            verbose (bool): whether to print output
        """
        with torch.no_grad():
            if verbose: print("Actor %d started." % self.rank)
            n = 0            

            env_out = self.env.initial(self.model_net)      
            if self.policy == PO_NET:      
                actor_state = self.actor_net.initial_state(batch_size=1)
                actor_out, _ = self.actor_net(env_out, actor_state)   
            elif self.policy == PO_MODEL:
                actor_out = self.po_model(env_out, self.model_net)
                action = actor_out.action
                actor_state = None

            train_actor = self.flags.train_actor and  self.policy == PO_NET and test_eps_n == 0
            train_model = self.flags.train_model and test_eps_n == 0

            # config for preloading before actor network start learning
            preload_needed = self.flags.train_model and self.flags.load_checkpoint
            preload = False            
            learner_actor_start = train_actor and (not preload_needed or preload)

            while (True):      
                # prepare train_actor_out data to be written
                initial_actor_state = actor_state
                if learner_actor_start:
                    self.write_actor_buffer(env_out, actor_out, 0)
                for t in range(self.flags.unroll_length):

                    # generate action
                    if self.policy == PO_NET:
                        # policy from applying actor network on the model-wrapped environment
                        actor_out, actor_state = self.actor_net(env_out, actor_state)    
                        action = [actor_out.action, actor_out.im_action, actor_out.reset_action]
                        if actor_out.term_action is not None:
                            action.append(actor_out.term_action)
                        action = torch.cat(action, dim=-1).unsqueeze(0)
                    elif self.policy == PO_MODEL:
                        # policy directly from the model
                        actor_out = self.po_model(env_out, self.model_net)
                        action = actor_out.action

                    env_out = self.env.step(action, self.employ_model_net)

                    # write the data to the respective buffers
                    if learner_actor_start:
                        self.write_actor_buffer(env_out, actor_out, t+1)       
                    if train_model and (self.policy != PO_NET or env_out.cur_t == 0): 
                        self.write_send_model_buffer(env_out, actor_out)   
                    if test_eps_n > 0:
                        finish, all_returns = self.write_test_buffer(
                            env_out, actor_out, test_eps_n, verbose)
                        if finish: return all_returns

                    #if torch.any(env_out.done):            
                    #    episode_returns = env_out.episode_return[env_out.done][:, 0]  
                    #    episode_returns = list(episode_returns.detach().cpu().numpy())
                    #    print(episode_returns)

                if learner_actor_start:
                    # send the data to remote actor buffer
                    train_actor_out = (self.actor_local_buffer, initial_actor_state)
                    status = 0
                    while (True):
                        status = ray.get(self.actor_buffer.get_status.remote())
                        if status == AB_FULL: time.sleep(0.01) 
                        else: break
                    if status == AB_FINISH: break
                    train_actor_out = ray.put(train_actor_out)
                    self.actor_buffer.write.remote([train_actor_out])

                # if preload buffer needed, check if preloaded
                if train_actor and preload_needed and not preload:
                    preload, tran_n = ray.get(self.model_buffer.check_preload.remote())
                    if self.rank == 0: 
                        if preload:
                            print("Finish preloading")
                            ray.get(self.model_buffer.set_preload.remote())
                        else:
                            print("Preloading: %d/%d" % (tran_n, self.flags.model_buffer_n))
                    learner_actor_start = not preload_needed or preload

                # update model weight                
                if n % 1 == 0:
                    if self.flags.train_actor and self.policy == PO_NET :
                        weights = ray.get(self.param_buffer.get_data.remote("actor_net"))
                        self.actor_net.set_weights(weights)
                    if self.flags.train_model:           
                        weights = ray.get(self.param_buffer.get_data.remote("model_net"))
                        self.model_net.set_weights(weights)                
                n += 1

        return True                          

    def write_actor_buffer(self, env_out: EnvOut, actor_out: ActorOut, t: int):
        # write local 
        if t == 0:
            fields = {}
            for field in TrainActorOut._fields:
                out = getattr(env_out if field in EnvOut._fields else actor_out, field)
                if out is not None:
                    fields[field] = torch.empty(size=(self.flags.unroll_length+1, 1)+out.shape[2:], dtype=out.dtype)
                    # each is in the shape of (T x B xdim_1 x dim_2 ...)
                else:
                    fields[field] = None
            self.actor_local_buffer = TrainActorOut(**fields)
            
        for field in TrainActorOut._fields:
            v = getattr(self.actor_local_buffer, field)            
            if v is not None:
                v[t] = getattr(env_out if field in EnvOut._fields else actor_out, field)[0,0]
        
        if t == self.flags.unroll_length:
            # post-processing
            self.actor_local_buffer = util.tuple_map(self.actor_local_buffer, lambda x: x.cpu())

    def empty_model_buffer(self):
        pre_shape = (self.flags.model_unroll_length + self.flags.model_k_step_return, 1,)
        return TrainModelOut(
            gym_env_out=torch.zeros(pre_shape + self.env.gym_env_out_shape, dtype=torch.uint8),
            policy_logits=torch.zeros(pre_shape + (self.env.num_actions,), dtype=torch.float32),
            action=torch.zeros(pre_shape, dtype=torch.long),
            reward=torch.zeros(pre_shape, dtype=torch.float32),
            done=torch.ones(pre_shape, dtype=torch.bool)) 

    def write_single_model_buffer(self, n: int, t: int, env_out: EnvOut, actor_out: ActorOut):
        self.model_local_buffer[n].gym_env_out[t] = env_out.gym_env_out[0,0]       
        self.model_local_buffer[n].reward[t] = env_out.reward[0,0,0]
        self.model_local_buffer[n].done[t] = env_out.done[0,0]
        self.model_local_buffer[n].policy_logits[t] = actor_out.policy_logits[0,0]
        self.model_local_buffer[n].action[t] = actor_out.action[0,0]

    def write_send_model_buffer(self, env_out: EnvOut, actor_out: ActorOut):
        n, t, cap_t, k = (self.model_n, self.model_t, self.flags.model_unroll_length,
            self.flags.model_k_step_return)
        self.write_single_model_buffer(n, t, env_out, actor_out)        

        if t >= cap_t:
            # write the beginning of another buffer
            self.write_single_model_buffer(1-n, t-cap_t, env_out, actor_out)

        if t >= cap_t + k - 1:
            # finish writing a buffer, send it
            self.model_buffer.write.remote(self.model_local_buffer[n])            
            self.model_local_buffer[n] = self.empty_model_buffer()
            self.model_n = 1 - n
            self.model_t = k
        else:
            self.model_t += 1

    def write_test_buffer(self, env_out: EnvOut, actor_out: ActorOut, 
        test_eps_n:int=0, verbose:bool=False):
        if torch.any(env_out.done):            
            episode_returns = env_out.episode_return[env_out.done][:, 0]  
            episode_returns = list(episode_returns.detach().cpu().numpy())
            for r in episode_returns:
                all_returns = ray.get(self.test_buffer.extend_data.remote("episode_returns", [r]))  
                all_returns = np.array(all_returns)         
                if verbose:       
                    print("%d Mean (Std.) : %.4f (%.4f) - %.4f" % (len(all_returns),
                        np.mean(all_returns), np.std(all_returns)/np.sqrt(len(all_returns)), r))
                if len(all_returns) > test_eps_n: return True, all_returns
                
        return False, None

    def po_model(self, env_out, model_net):
        _, _, policy_logits, _ = model_net(env_out.gym_env_out[0], env_out.last_action[:,:,0], one_hot=False)                        
        action = torch.multinomial(F.softmax(policy_logits[0], dim=1), num_samples=1).unsqueeze(0)
        actor_out = util.construct_tuple(ActorOut, policy_logits=policy_logits, action=action)        
        return actor_out