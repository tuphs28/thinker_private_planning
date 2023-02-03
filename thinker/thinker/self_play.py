import time, timeit
import numpy as np
import argparse
from collections import namedtuple
import traceback
import torch
from torch import nn
from torch.nn import functional as F
import ray
from thinker.buffer import ActorBuffer, ModelBuffer, GeneralBuffer
from thinker.buffer import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.net import ActorNet, ModelNet, ActorOut
from thinker.env import Environment, EnvOut
import thinker.util as util

#_fields = tuple(item for item in ActorOut._fields + EnvOut._fields if item != 'gym_env_out')
_fields = tuple(item for item in ActorOut._fields + EnvOut._fields)
TrainActorOut = namedtuple('TrainActorOut', _fields)
TrainModelOut = namedtuple('TrainModelOut', ['gym_env_out', 'policy_logits', 'action', 'reward', 'done', 'baseline'])

PO_NET, PO_MODEL, PO_NSTEP = 0, 1, 2

@ray.remote
class SelfPlayWorker():
    def __init__(self, param_buffer:GeneralBuffer, actor_buffer: ActorBuffer, 
            model_buffer:ModelBuffer, test_buffer:GeneralBuffer,
            policy:int, policy_params:dict, rank: int, num_p_actors: int, flags:argparse.Namespace):                        
        self._logger = util.logger()
        if not flags.disable_cuda and torch.cuda.is_available() and num_p_actors > 1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self._logger.info("Initalizing actor %d with device %s %s" % (
            rank, "cuda" if self.device == torch.device("cuda") else "cpu",
            "(test mode)" if test_buffer is not None else ""))

        self.param_buffer = param_buffer
        self.actor_buffer = actor_buffer
        self.model_buffer = model_buffer
        self.test_buffer = test_buffer  
        self.policy = policy
        self.policy_params = policy_params
        self.rank = rank
        self.num_p_actors = num_p_actors
        self.flags = flags
        self.timing = util.Timings()

        self.env = Environment(flags, model_wrap=policy==PO_NET, env_n=num_p_actors, device=self.device)
        seed = [1 + i + num_p_actors * rank for i in range(num_p_actors)]
        self.env.seed(seed)

        if self.policy==PO_NET:
            self.actor_net = ActorNet(obs_shape=self.env.model_out_shape, 
                                      gym_obs_shape=self.env.gym_env_out_shape,
                                      num_actions=self.env.num_actions, 
                                      flags=flags)
            self.actor_net.to(self.device)

        self.model_net = ModelNet(obs_shape=self.env.gym_env_out_shape, num_actions=self.env.num_actions, flags=flags)
        self.model_net.train(False)
        self.model_net.to(self.device)


        # the networks weight are set by the respective learner; but if the respective
        # learner does not exist, then rank 0 worker will set the weights
        if rank == 0 and not self.flags.train_actor and self.policy==PO_NET:
            if self.flags.preload_actor:
                checkpoint = torch.load(self.flags.preload_actor, map_location=torch.device('cpu'))
                self.actor_net.set_weights(checkpoint["actor_net_state_dict"])  
                self._logger.info("Loadded actor network from %s" % self.flags.preload_actor)
            self.param_buffer.set_data.remote("actor_net", self.actor_net.get_weights())

        if rank == 0 and not self.flags.train_model:
            if self.flags.preload_model:
                checkpoint = torch.load(self.flags.preload_model, map_location=torch.device('cpu'))
                self.model_net.set_weights(checkpoint["model_state_dict"] if "model_state_dict" in 
                    checkpoint else checkpoint["model_net_state_dict"])  
                self._logger.info("Loadded model network from %s" % self.flags.preload_model)
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
                self._logger.info("Override model network from %s" % self.flags.employ_model)
        else:
            self.employ_model_net = self.model_net

        if flags.train_model: 
            self.model_local_buffer = [self.empty_model_buffer(), self.empty_model_buffer()]        
            self.model_n = 0
            self.model_t = 0
            if self.flags.model_rnn:
                self.initial_model_state = self.model_net.init_state(self.num_p_actors, device=self.device)
                self.model_state = self.model_net.init_state(self.num_p_actors, device=self.device)                        

    def gen_data(self, test_eps_n:int=0, verbose:bool=True):
        """ Generate self-play data
        Args:
            test_eps_n (int): number of episode to test for (only for testing mode);
            if set to non-zero, the worker will stop once reaching test_eps_n episodes
            and the data will not be sent out to model or actor buffer
            verbose (bool): whether to print output
        """
        try:
            with torch.no_grad():
                if verbose: self._logger.info("Actor %d started. %s" % (self.rank, "(test mode)" if test_eps_n > 0 else ""))
                n = 0            

                if self.policy == PO_NET:
                    env_out, self.employ_model_state = self.env.initial(self.employ_model_net)      
                else:
                    env_out = self.env.initial(self.employ_model_net)   
                if self.policy == PO_NET:                          
                    actor_state = self.actor_net.initial_state(batch_size=self.num_p_actors, device=self.device)                    
                    actor_out, _ = self.actor_net(env_out, actor_state)   
                elif self.policy == PO_MODEL:
                    actor_out = self.po_model(env_out, self.model_net)
                    action = actor_out.action
                    actor_state = None
                elif self.policy == PO_NSTEP:
                    actor_out = self.po_nstep(self.env, self.model_net)
                    action = actor_out.action
                    actor_state = None                

                train_actor = self.flags.train_actor and self.policy == PO_NET and test_eps_n == 0
                train_model = self.flags.train_model and test_eps_n == 0

                # config for preloading before actor network start learning
                preload_needed = self.flags.train_model and self.flags.load_checkpoint
                preload = False            
                learner_actor_start = train_actor and (not preload_needed or preload)

                timer = timeit.default_timer
                start_time = timer()

                while (True):      
                    if self.rank == 0: self.timing.reset()
                    # prepare train_actor_out data to be written
                    initial_actor_state = actor_state
                    if learner_actor_start:
                        self.write_actor_buffer(env_out, actor_out, 0)
                    if self.rank == 0: self.timing.time("misc1")
                    for t in range(self.flags.unroll_length):
                        # generate action
                        if self.policy == PO_NET:
                            # policy from applying actor network on the model-wrapped environment
                            if self.flags.float16 and self.device == torch.device("cuda"):
                                with torch.autocast(device_type='cuda', dtype=torch.float16):    
                                    actor_out, actor_state = self.actor_net(env_out, actor_state)    
                            else:
                                actor_out, actor_state = self.actor_net(env_out, actor_state)    
                            if self.rank == 0: self.timing.time("actor_net")
                            action = [actor_out.action, actor_out.im_action, actor_out.reset_action]
                            if actor_out.term_action is not None:
                                action.append(actor_out.term_action)
                            action = torch.cat([a.unsqueeze(-1) for a in action], dim=-1)     
                        elif self.policy == PO_MODEL:
                            # policy directly from the model
                            actor_out = self.po_model(env_out, self.model_net)
                            action = actor_out.action
                        elif self.policy == PO_NSTEP:
                            actor_out = self.po_nstep(self.env, self.model_net)
                            action = actor_out.action
                        if self.policy == PO_NET:
                            if self.flags.float16 and self.device == torch.device("cuda"):
                                with torch.autocast(device_type='cuda', dtype=torch.float16): 
                                    env_out, self.employ_model_state = self.env.step(action, self.employ_model_net,
                                        self.employ_model_state)
                            else:
                                env_out, self.employ_model_state = self.env.step(action, self.employ_model_net,
                                        self.employ_model_state)
                            if self.rank == 0: self.timing.time("step env")
                        else:
                            env_out = self.env.step(action, self.employ_model_net)
                        # write the data to the respective buffers
                        if learner_actor_start:
                            self.write_actor_buffer(env_out, actor_out, t+1)       
                        if self.rank == 0: self.timing.time("write_actor_buffer")
                        if train_model and (self.policy != PO_NET or env_out.cur_t[:,0] == 0): 
                            baseline = None
                            if self.policy == PO_NET:
                                if self.flags.model_bootstrap_type == 0:
                                    baseline = self.env.env.baseline_mean_q    
                                elif self.flags.model_bootstrap_type == 1:
                                    baseline = self.env.env.baseline_max_q
                                elif self.flags.model_bootstrap_type == 2:
                                    baseline = actor_out.baseline[:, :, 0] / (self.flags.discounting ** ((self.flags.rec_t - 1)/ self.flags.rec_t))
                            self.write_send_model_buffer(env_out, actor_out, baseline)   
                        if self.rank == 0: self.timing.time("write_send_model_buffer")
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
                        if self.device != torch.device("cpu"):
                            initial_actor_state = util.tuple_map(initial_actor_state, lambda x: x.cpu())
                        train_actor_out = (self.actor_local_buffer, initial_actor_state)
                        status = 0
                        if self.rank == 0: self.timing.time("mics2")
                        while (True):
                            status = ray.get(self.actor_buffer.get_status.remote())
                            if status == AB_FULL: 
                                time.sleep(0.1) 
                            else: 
                                break
                        if status == AB_FINISH: break
                        train_actor_out = ray.put(train_actor_out)
                        self.actor_buffer.write.remote([train_actor_out])
                        if self.rank == 0: self.timing.time("send actor_buffer")
                    # if preload buffer needed, check if preloaded
                    if train_actor and preload_needed and not preload:
                        preload, tran_n = ray.get(self.model_buffer.check_preload.remote())
                        if self.rank == 0: 
                            if preload:
                                self._logger.info("Finish preloading")
                                ray.get(self.model_buffer.set_preload.remote())
                            else:
                                self._logger.info("Preloading: %d/%d" % (tran_n, self.flags.model_buffer_n))
                        learner_actor_start = not preload_needed or preload
                    if self.rank == 0: self.timing.time("mics3")                    
                    # update model weight                
                    if n % 10000 == 0:
                        if self.flags.train_actor and self.policy == PO_NET :
                            weights = ray.get(self.param_buffer.get_data.remote("actor_net"))
                            self.actor_net.set_weights(weights)
                        if self.flags.train_model:           
                            weights = ray.get(self.param_buffer.get_data.remote("model_net"))
                            self.model_net.set_weights(weights)     
                    if self.rank == 0: self.timing.time("update model weight")                    
                    # Signal control for all self-play threads (only when it is not in testing mode)
                    if test_eps_n == 0:
                        signals = ray.get(self.param_buffer.get_data.remote("self_play_signals"))
                        while (signals is not None and "halt" in signals and signals["halt"]):
                            time.sleep(0.1)
                            signals = ray.get(self.param_buffer.get_data.remote("self_play_signals"))
                        if (signals is not None and "term" in signals and signals["term"]):
                            return True               
                    if self.rank == 0: self.timing.time("signal control")                                                
                    n += 1
                    if self.rank == 0 and timer() - start_time > 5: 
                        self._logger.info(self.timing.summary())
                        start_time = timer()

        except:
            # printing stack trace
            traceback.print_exc()
            return False
        return True                          

    def write_actor_buffer(self, env_out: EnvOut, actor_out: ActorOut, t: int):
        # write local 
        if t == 0:
            fields = {}
            for field in TrainActorOut._fields:
                out = getattr(env_out if field in EnvOut._fields else actor_out, field)
                if out is not None and (self.flags.actor_see_p > 0 or field != "gym_env_out"):
                    fields[field] = torch.empty(size=(self.flags.unroll_length+1, self.num_p_actors)+out.shape[2:], 
                        dtype=out.dtype, device=self.device)
                    # each is in the shape of (T x B xdim_1 x dim_2 ...)
                else:
                    fields[field] = None
            self.actor_local_buffer = TrainActorOut(**fields)
            
        for field in TrainActorOut._fields:
            v = getattr(self.actor_local_buffer, field)            
            if v is not None:
                v[t] = getattr(env_out if field in EnvOut._fields else actor_out, field)[0]
        
        if t == self.flags.unroll_length:
            # post-processing
            self.actor_local_buffer = util.tuple_map(self.actor_local_buffer, lambda x: x.cpu().numpy())

    def empty_model_buffer(self):
        pre_shape = (self.flags.model_unroll_length + 2 * self.flags.model_k_step_return, self.num_p_actors,)
        return TrainModelOut(
            gym_env_out=torch.zeros(pre_shape + self.env.gym_env_out_shape, dtype=torch.uint8, device=self.device),
            policy_logits=torch.zeros(pre_shape + (self.env.num_actions,), dtype=torch.float32, device=self.device),
            action=torch.zeros(pre_shape, dtype=torch.long, device=self.device),
            reward=torch.zeros(pre_shape, dtype=torch.float32, device=self.device),
            done=torch.ones(pre_shape, dtype=torch.bool, device=self.device),
            baseline=torch.zeros(pre_shape, dtype=torch.float32, device=self.device)) 

    def write_single_model_buffer(self, n: int, t: int, env_out: EnvOut, actor_out: ActorOut, baseline:torch.tensor):
        self.model_local_buffer[n].gym_env_out[t] = env_out.gym_env_out[0]       
        self.model_local_buffer[n].reward[t] = env_out.reward[0,:,0]
        self.model_local_buffer[n].done[t] = env_out.done[0]
        self.model_local_buffer[n].policy_logits[t] = actor_out.policy_logits[0]
        self.model_local_buffer[n].action[t] = actor_out.action[0]
        if baseline is not None:
            self.model_local_buffer[n].baseline[t] = baseline

    def write_send_model_buffer(self, env_out: EnvOut, actor_out: ActorOut, baseline:torch.tensor):
        n, t, cap_t, k = (self.model_n, self.model_t, self.flags.model_unroll_length,
            self.flags.model_k_step_return)
        self.write_single_model_buffer(n, t, env_out, actor_out, baseline)        

        if self.flags.model_rnn:   
            # if the learnt model is rnn but not the same as employ_model, 
            # we have to compute the model state
            if self.flags.employ_model:
                with torch.no_grad:
                    _, _, self.model_state = self.model_net(
                        x=env_out.gym_env_out, 
                        actions=env_out.last_action[:, :, 0], 
                        done=env_out.done,
                        state=self.model_state, 
                        one_hot=False)
            else:
                self.model_state = self.employ_model_state
                    
            if t == cap_t - 1:
                self.initial_model_state_ = self.model_state

        if t >= cap_t:
            # write the beginning of another buffer
            self.write_single_model_buffer(1-n, t-cap_t, env_out, actor_out, baseline)      

        if t >= cap_t + 2 * k - 2:
            # finish writing a buffer, send it
            if self.device != torch.device("cpu"):
                send_model_local_buffer = util.tuple_map(self.model_local_buffer[n], lambda x: x.cpu())
            else:
                send_model_local_buffer = self.model_local_buffer[n]
            self.model_buffer.write.remote(send_model_local_buffer, 
                self.initial_model_state if self.flags.model_rnn else None, 
                self.rank)            
            if self.flags.model_rnn: 
                self.initial_model_state = self.initial_model_state_ 
            self.model_local_buffer[n] = self.empty_model_buffer()
            self.model_n = 1 - n
            self.model_t = t - cap_t + 1
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
                    self._logger.info("%d Mean (Std.) : %.4f (%.4f) - %.4f" % (len(all_returns),
                        np.mean(all_returns), np.std(all_returns)/np.sqrt(len(all_returns)), r))
                if len(all_returns) > test_eps_n: return True, all_returns
                
        return False, None

    def po_model(self, env_out, model_net):
        if not model_net.rnn:
            _, _, policy_logits, _ = model_net(env_out.gym_env_out[0], env_out.last_action[:,:,0], one_hot=False)                        
        else:
            if not hasattr(self, 'model_state'): 
                self.model_state = self.model_net.init_state(bsz=1)
            _, policy_logits, self.model_state = model_net(x=env_out.gym_env_out, 
                actions=env_out.last_action[:,:,0], 
                done=env_out.done,                
                state=self.model_state,
                one_hot=False,)                        

        action = torch.multinomial(F.softmax(policy_logits[0], dim=1), num_samples=1).unsqueeze(0)
        actor_out = util.construct_tuple(ActorOut, policy_logits=policy_logits, action=action)        
        # policy_logits has shape (T, B, num_actions)
        # action has shape (T, B, 1)
        return actor_out

    def po_nstep(self, env, model_net):
        discounting = self.flags.discounting        
        if self.policy_params is not None:
            n = self.policy_params["n"]
            temp = self.policy_params["temp"]
        else:
            n, temp = 1, 0.5 # default policy param
        policy_logits, action, _ = self.nstep(env, model_net, discounting, n, temp)
        policy_logits = policy_logits.unsqueeze(0)
        action = action.unsqueeze(0)
        actor_out = util.construct_tuple(ActorOut, policy_logits=policy_logits, action=action)               
        return actor_out 

    def nstep(self, env, model_net, discounting, n, temp):  
        with torch.no_grad():
            num_actions = env.num_actions
            q_ret = torch.zeros(1, num_actions)    
            state = env.clone_state()
            for act in range(num_actions):
                env_out = env.step(torch.full(size=(1, 1, 1), fill_value=act, dtype=torch.long))
                if n > 1:
                    _, _, sub_q_ret = self.nstep(env, model_net, discounting, n-1, temp)
                    ret = env_out.reward + discounting * torch.max(sub_q_ret, dim=1)[0] * (~env_out.done).float()
                else:
                    _, baseline, _, _ = model_net(env_out.gym_env_out[0], env_out.last_action[:,:,0], one_hot=False)
                    ret = env_out.reward + discounting * baseline * (~env_out.done).float()
                q_ret[:, act] = ret
                env.restore_state(state)

            policy_logits = q_ret / temp
            prob = F.softmax(policy_logits, dim=1)
            action = torch.multinomial(prob, num_samples=1)
        return policy_logits, action, q_ret  