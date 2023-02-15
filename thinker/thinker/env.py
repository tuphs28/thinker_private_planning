from collections import namedtuple, deque
import numpy as np
import cv2
import torch
from torch.nn import functional as F
import gym
from thinker.gym_add.asyn_vector_env import AsyncVectorEnv
from thinker import util

EnvOut = namedtuple('EnvOut', ['gym_env_out', 'model_out', 'model_encodes', 'see_mask', 'reward', 'done', 'real_done',
    'truncated_done', 'episode_return', 'episode_step', 'cur_t', 'last_action', 'max_rollout_depth'])

def Environment(flags, model_wrap=True, env_n=1, device=None, time=False):
    """Create an environment using flags.env; first
    wrap the env with basic wrapper, then wrap it
    with a model, and finally wrap it with PosWrapper
    for episode processing

    Args:
        flags (namespace): configuration flags
        model_wrap (boolean): whether to wrap the environment with model
        env_n (int): number of parallel env
        device (torch.device): device that runs the model 
    Return:
        environment
    """
    if flags.env == "Sokoban-v0": import gym_sokoban
    if flags.env == "cSokoban-v0": import gym_csokoban
    if flags.cwrapper: from thinker.cenv import cVecModelWrapper

    if env_n == 1:
        if model_wrap:
            env = PostWrapper(ModelWrapper(PreWrap(gym.make(flags.env), flags.env), flags), True, flags)
        else:
            env = PostWrapper(PreWrap(gym.make(flags.env), flags.env), False, flags)    
    else:
        if model_wrap:
            env = AsyncVectorEnv([lambda: PreWrap(gym.make(flags.env), flags.env) for _ in range(env_n)])
            num_actions = env.action_space[0].n
            uVecModelWrapper = cVecModelWrapper if flags.cwrapper else VecModelWrapper
            env = PostVecModelWrapper(uVecModelWrapper(env, env_n, flags, device=device, time=time), env_n, num_actions, flags, device=device)
        else:
            raise Exception("Parallel run only supports for not model_wrap environments")
    return env

def PreWrap(env, name):
    if name == "cSokoban-v0":
        env = TransposeWrap(env)
    elif name == "Sokoban-v0":
        env = NoopWrapper(env, cost=-0.01)
        env = WarpFrame(env, width=80, height=80, grayscale=False)
        env = TimeLimit_(env, max_episode_steps=120)
        env = TransposeWrap(env)
    else:        
        env = StateWrapper(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = wrap_deepmind(env,
            episode_life=True,
            clip_rewards=False,
            frame_stack=True,
            scale=False,)
        env = TransposeWrap(env)
    return env

def _format(gym_env_out, model_out):
    """Add batch and time index to env output"""
    if gym_env_out is not None:
        gym_env_out = torch.from_numpy(gym_env_out)
        gym_env_out = gym_env_out.view((1, 1) + gym_env_out.shape)        
    else:
        gym_env_out = None
    if model_out is not None:
        model_out = model_out.view((1, 1) + model_out.shape)
    else:
        model_out = None
    return gym_env_out, model_out

class PostWrapper:
    """The final wrapper for env.; convert all return to tensor and 
    calculates the episode return"""

    def __init__(self, env, model_wrap, flags):
        self.env = env
        self.flags = flags
        self.episode_return = None
        self.episode_step = None
        self.model_wrap = model_wrap
        self.num_actions = env.env.action_space.n 
        self.actor_see_p = flags.actor_see_p

        if self.model_wrap:
            self.model_out_shape = env.observation_space.shape
            self.gym_env_out_shape = env.env.observation_space.shape                
        else:
            self.gym_env_out_shape = env.observation_space.shape                

    def initial(self, model_net=None):
        if self.model_wrap:
            reward_shape = 2 if self.flags.reward_type == 1 else 1
            action_shape = 4 if self.flags.flex_t else 3
        else:
            reward_shape = 1
            action_shape = 1

        initial_reward = torch.zeros(1, 1, reward_shape)
        # This supports only single-tensor actions.
        self.last_action = torch.zeros(1, 1, action_shape, dtype=torch.long)
        self.episode_return = torch.zeros(1, 1, reward_shape)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        initial_real_done = torch.ones(1, 1, dtype=torch.bool)

        if self.model_wrap:
            out, model_state = self.env.reset(model_net)
            gym_env_out, model_out = _format(*out)
        else:
            gym_env_out, model_out = _format(self.env.reset(), None)
        see_mask = torch.rand(size=(1, 1)) > (1 - self.actor_see_p)

        ret = EnvOut(
            gym_env_out=gym_env_out,
            model_out=model_out,
            model_encodes=None,
            see_mask=see_mask,
            reward=initial_reward,
            done=initial_done,
            real_done=initial_real_done,
            truncated_done=torch.tensor(0).view(1, 1).bool(),
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            cur_t=torch.tensor(0).view(1, 1) if self.model_wrap else None,
            last_action=self.last_action,
            max_rollout_depth=torch.tensor(0.).view(1, 1) if self.model_wrap else None
        )
        return (ret, model_state) if self.model_wrap else ret

    def step(self, action, model_net=None, model_state=None):
        assert len(action.shape) == 3, "dim of action should be 3"
        if self.model_wrap:            
            out, reward, done, info, model_state = self.env.step(action[0,0].cpu().detach().numpy(), model_net, model_state)     
            gym_env_out, model_out = _format(*out)
        else:
            out, reward, done, info = self.env.step(action[0,0,0].cpu().detach().numpy())     
            gym_env_out, model_out = _format(out, None)
        self.episode_step += 1
        self.episode_return = self.episode_return + torch.tensor(reward).unsqueeze(0).unsqueeze(0)
        episode_step = self.episode_step
        episode_return = self.episode_return.clone()
        if done:
            if self.model_wrap:
                out, model_state = self.env.reset(model_net)
                gym_env_out, model_out = _format(*out)
            else:
                gym_env_out, model_out = _format(self.env.reset(), None)

        real_done = info["real_done"] if "real_done" in info else done
        if real_done:
            self.episode_return = torch.zeros(1, 1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)     
        real_done = torch.tensor(real_done).view(1, 1)
        
        reward = torch.tensor(reward).view(1, 1, -1)
        done = torch.tensor(done).view(1, 1)
        truncated_done = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        truncated_done = torch.tensor(truncated_done).view(1, 1)
        if self.model_wrap:
            cur_t = torch.tensor(info["cur_t"]).view(1, 1)
            if cur_t == 0 and self.episode_return.shape[2] > 1:
                self.episode_return[:, :, 1] = 0.
            if 'max_rollout_depth' in info:
                max_rollout_depth = torch.tensor(info["max_rollout_depth"]).view(1, 1)
            else:
                max_rollout_depth = torch.tensor(0.).view(1, 1)
        else:
            cur_t, max_rollout_depth = None, None

        if self.model_wrap:
            if cur_t == 0:
                self.last_action = action
            else:
                self.last_action = self.last_action.clone()            
                self.last_action[:, :, 1:] = action[:, :, 1:]
        else:
            self.last_action = action
        see_mask = torch.rand(size=(1, 1)) > (1 - self.actor_see_p)

        if self.flags.reward_clipping > 0:
            reward = torch.clamp(reward, - self.flags.reward_clipping, self.flags.reward_clipping)

        ret = EnvOut(
            gym_env_out=gym_env_out,
            model_out=model_out,
            model_encodes=None,
            see_mask=see_mask,
            reward=reward,
            done=done,
            real_done=real_done,
            truncated_done=truncated_done,          
            episode_return=episode_return,
            episode_step=episode_step,
            cur_t=cur_t,
            last_action=self.last_action,
            max_rollout_depth=max_rollout_depth
        )
        return (ret, model_state) if self.model_wrap else ret

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed[0])    

    def clone_state(self):
        state = self.env.clone_state()
        state["env_episode_return"] = self.episode_return.clone()
        state["env_episode_step"] = self.episode_step.clone()
        return state
        
    def restore_state(self, state):
        self.episode_return = state["env_episode_return"].clone()
        self.episode_step = state["env_episode_step"].clone()
        self.env.restore_state(state)

class ModelWrapper(gym.Wrapper):
    """Wrap the gym environment with a model; output for each 
    step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out) that corresponds to underlying 
    environment frame and output from the model wrapper.
    """

    def __init__(self, env, flags):
        gym.Wrapper.__init__(self, env)
        
        self.env = env     
        self.rec_t = flags.rec_t        
        self.flex_t = flags.flex_t 
        self.flex_t_cost = flags.flex_t_cost         
        self.discounting = flags.discounting
        self.depth_discounting = flags.depth_discounting
        self.reward_type = flags.reward_type    
        self.reward_transform = flags.reward_transform
        self.no_mem = flags.no_mem
        self.perfect_model = flags.perfect_model
        self.reset_m = flags.reset_m
        self.tree_carry = flags.tree_carry
        self.thres_carry = flags.thres_carry        
        self.thres_discounting = flags.thres_discounting        
        self.num_actions = env.action_space.n
        self.cur_node = None
        self.root_node = None
        self.ver = 0 if not hasattr(flags, "model_wrapper_ver") else flags.model_wrapper_ver
            
        if not self.flex_t:
            obs_n = 9 + self.num_actions * 10 + self.rec_t
        else:
            obs_n = 11 + self.num_actions * 10 
        
        self.observation_space = gym.spaces.Box(
          low=-np.inf, high=np.inf, shape=(obs_n, 1, 1), dtype=float)
        
        self.max_rollout_depth = 0.
        self.rollout_depth = 0.
        self.thres = None
        self.baseline_max_q = torch.zeros(1, dtype=torch.float32)
        self.baseline_mean_q = torch.zeros(1, dtype=torch.float32)
        self.root_max_q = None        
        
    def reset(self, model_net, **kwargs):
        x = self.env.reset(**kwargs)
        self.cur_t = 0        
        self.rollout_depth = 0.    
        model_state = model_net.init_state(1) if model_net.rnn else None                
        out, _, model_state = self.use_model(model_net=model_net, 
            model_state = model_state, x=x, r=0.,
            a=0, cur_t=self.cur_t, reset=1., term=0., done=False)
        if self.reward_type == 1:
            self.last_root_max_q = self.root_max_q
        return (x, out), model_state
    
    def step(self, action, model_net, model_state=None):  
        if not self.flex_t:
            re_action, im_action, reset = action
            term = None
        else:
            re_action, im_action, reset, term = action
        info = {}
        info["max_rollout_depth"] = self.max_rollout_depth
        if (not self.flex_t and self.cur_t < self.rec_t - 1) or (
            self.flex_t and self.cur_t < self.rec_t - 1 and not term):
          # imagainary step
          self.cur_t += 1
          out, x, model_state = self.use_model(model_net=model_net, 
            model_state=None, x=None, r=None, a=im_action, 
            cur_t=self.cur_t, reset=reset, term=term, done=False)   # use the imagine x in imagine step       
          if self.reward_type == 0:
            r = np.array([0.])
          else:
            if self.flex_t:
                flex_t_cost = self.flex_t_cost
            else:                
                flex_t_cost = 0.
            if self.depth_discounting < 1.:
                dc = self.depth_discounting ** self.last_rollout_depth
            else:
                dc = 1.
            r = np.array([0., ((self.root_max_q - self.last_root_max_q)*dc - flex_t_cost).item()], dtype=np.float32)
          done = False
          info['cur_t'] = self.cur_t   
        else:
          # real step
          # record the root stat of previous state before move forward
          rollout_qs = self.root_node.rollout_qs 
          self.baseline_mean_q = torch.mean(torch.concat(rollout_qs)).unsqueeze(-1) / self.discounting
          self.baseline_max_q = torch.max(torch.concat(rollout_qs)).unsqueeze(-1) / self.discounting

          self.cur_t = 0
          if self.perfect_model: self.env.restore_state(self.root_node.encoded["env_state"])
          x, r, done, info_ = self.env.step(re_action)                    
          out, _, model_state = self.use_model(model_net=model_net, 
            model_state=model_state, x=x, r=r, a=re_action, 
            cur_t=self.cur_t, reset=1., term=term, done=done) # use the real x in real step
          info.update(info_)
          info['cur_t'] = self.cur_t
          if self.reward_type == 0:
            r = np.array([r])
          else:
            r = np.array([r, 0.], dtype=np.float32)   
        if self.reward_type == 1:          
            self.last_root_max_q = self.root_max_q   
        
        return (x, out), r, done, info, model_state
        
    def use_model(self, model_net, model_state, x, r, a, cur_t, reset, term, done):     
        with torch.no_grad():
            self.last_rollout_depth = self.rollout_depth

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
                else:
                    re_action = a                             
                
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # x is in shape (1, C, H, W)
                self.x = self.x_ = x_tensor
                a_tensor = F.one_hot(torch.tensor(re_action, dtype=torch.long).unsqueeze(0), self.num_actions) 
                # a_tensor is in shape (1, num_action,)
                if not model_net.rnn:                    
                    _, vs, _, logits, encodeds = model_net(x_tensor, a_tensor.unsqueeze(0), one_hot=True)                                         
                else:
                    vs, logits, model_state = model_net(x=x_tensor.unsqueeze(0), 
                              actions=a_tensor.unsqueeze(0), 
                              done=torch.tensor(done, dtype=bool).unsqueeze(0).unsqueeze(0),
                              state=model_state,
                              one_hot=True) 
                self._debug = (x_tensor,a_tensor.unsqueeze(0), logits, x)
                
                if self.perfect_model: 
                    encoded = {"env_state": self.clone_state()}
                else:
                    encoded=encodeds[-1]
                encoded["x"] = x
                if model_net.rnn: encoded["model_state"] = model_state
                
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
                    self.root_node.parent = None
                
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
                        rs, vs, logits, encodeds = model_net.forward_encoded(self.cur_node.encoded, 
                            a_tensor.unsqueeze(0), one_hot=True)
                        next_node.expand(r=rs[-1, 0].unsqueeze(-1), v=vs[-1, 0].unsqueeze(-1), 
                                     logits=logits[-1, 0], encoded=encodeds[-1])
                    else:                        
                        if "done" not in self.cur_node.encoded:                            
                            self.env.restore_state(self.cur_node.encoded["env_state"])                        
                            x, r, done, info = self.env.step(a) 
                            encoded = {"env_state": self.clone_state()}
                            encoded["x"] = x
                            if done: encoded["done"] = True                        
                            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                            self.x_ = x_tensor
                            a_tensor = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions) 

                            if not model_net.rnn:
                                _, vs, _, logits, encodeds = model_net(x_tensor, a_tensor.unsqueeze(0), one_hot=True)                     
                            else:
                                vs, logits, model_state = model_net(x=x_tensor.unsqueeze(0), 
                                        actions=a_tensor.unsqueeze(0), 
                                        done=torch.tensor(done, dtype=bool).unsqueeze(0).unsqueeze(0),
                                        state=self.cur_node.encoded["model_state"],
                                        one_hot=True) 
                                encoded["model_state"] = model_state      

                            if done:
                                v = torch.tensor([0.], dtype=torch.float32)
                            else:
                                v = vs[-1, 0].unsqueeze(-1)

                            next_node.expand(r=torch.tensor([r], dtype=torch.float32), 
                                             v=v, 
                                             logits=logits[-1, 0], 
                                             encoded=encoded)
                        else:
                            logits = torch.concat([ch.logit for ch in self.cur_node.children])  
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
            
            if self.ver > 0:
                root_node_stat = self.root_node.stat(detailed=True, reward_transform=self.reward_transform)
            else:
                root_node_stat = self.root_node.stat(detailed=False, reward_transform=self.reward_transform)
            cur_node_stat = self.cur_node.stat(detailed=False, reward_transform=self.reward_transform)                        
            reset = torch.tensor([reset], dtype=torch.float32)            
            depc = torch.tensor([self.discounting ** (self.rollout_depth)])
            
            root_trail_r = self.root_node.trail_r / self.discounting
            root_rollout_q = self.root_node.rollout_q / self.discounting
            rollout_qs = self.root_node.rollout_qs
            root_max_q = torch.max(torch.concat(rollout_qs)).unsqueeze(-1) / self.discounting
            if self.thres_carry and self.thres is not None:
                root_max_q = torch.max(root_max_q, self.thres)
                
            if not self.flex_t:
                time = F.one_hot(torch.tensor(cur_t).long(), self.rec_t)
            else:
                time = torch.tensor([self.discounting ** (self.cur_t)])                    
                
            if not self.flex_t:
                if self.ver > 0:
                    ret_list = [root_node_stat, cur_node_stat, reset, time, depc]
                else:                    
                    ret_list = [root_node_stat, cur_node_stat, reset, time, depc, root_trail_r, root_rollout_q, root_max_q]
            else:
                term = torch.tensor([term], dtype=torch.float32)                            
                if self.ver > 0:
                    ret_list = [root_node_stat, cur_node_stat, reset, term, time, depc]
                else:
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
            x = self.cur_node.encoded["x"]
            
            if self.thres is not None:
                self.ret_dict["thres"] = self.thres
            
            if reset:
                self.rollout_depth = 0
                self.unexpand_rollout_depth = 0.
                self.cur_node = self.root_node
                self.cur_node.visit()
                self.pass_unexpand = False
            
            return out, x, self.root_node.encoded["model_state"] if model_net.rnn else None

    def seed(self, seed):
        self.env.seed(seed)


class PostVecModelWrapper(gym.Wrapper):
    """The final wrapper for env.; calculates episode return,
       record last action, and returns EnvOut"""

    def __init__(self, env, env_n, num_actions, flags, device=None):
        self.device = torch.device("cpu") if device is None else device
        self.env = env
        self.env_n = env_n
        self.flags = flags
        self.num_actions = num_actions
        self.actor_see_p = flags.actor_see_p 
        self.actor_see_encode = flags.actor_see_encode
        self.model_out_shape = env.model_out_shape
        self.gym_env_out_shape = env.gym_env_out_shape
        self.reward_type = flags.reward_type

    def initial(self, model_net):
        reward_shape = 2 if self.flags.reward_type == 1 else 1
        action_shape = 4 if self.flags.flex_t else 3
        self.last_action = torch.zeros(self.env_n, action_shape, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(1, self.env_n, reward_shape, dtype=torch.float32, device=self.device)
        self.episode_step = torch.zeros(1, self.env_n, dtype=torch.long, device=self.device)

        model_out, gym_env_out, model_encodes = self.env.reset(model_net) 
        model_out = model_out.unsqueeze(0)
        gym_env_out = gym_env_out.unsqueeze(0)
        if self.actor_see_encode: model_encodes = model_encodes.unsqueeze(0)

        ret = EnvOut(
            gym_env_out=gym_env_out,
            model_out=model_out,
            model_encodes=model_encodes,
            see_mask=torch.rand(size=(1, self.env_n), device=self.device) > (1 - self.actor_see_p),
            reward=torch.zeros(1, self.env_n, reward_shape, dtype=torch.float32, device=self.device),
            done=torch.ones(1, self.env_n, dtype=torch.bool, device=self.device),
            real_done=torch.ones(1, self.env_n, dtype=torch.bool, device=self.device),
            truncated_done=None,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            cur_t=torch.zeros(1, self.env_n, dtype=torch.long, device=self.device),
            last_action=self.last_action.unsqueeze(0),
            max_rollout_depth=torch.zeros(1, self.env_n, dtype=torch.long, device=self.device)
        )
        return ret, None

    def step(self, action, model_net=None, model_state=None):
        action_shape = 4 if self.flags.flex_t else 3
        assert action.shape == (1, self.env_n, action_shape), (
            "shape of action should be (1, B, %d)" % action_shape)
        out, reward, done, info = self.env.step(action[0], model_net)     
        real_done = info["real_done"]
        model_out, gym_env_out, model_encodes = out
        model_out = model_out.unsqueeze(0)
        gym_env_out = gym_env_out.unsqueeze(0)
        if self.actor_see_encode: model_encodes = model_encodes.unsqueeze(0)

        self.episode_step += 1
        self.episode_return = self.episode_return + reward.unsqueeze(0)

        episode_step = self.episode_step.clone()
        episode_return = self.episode_return.clone()

        cur_t = info["cur_t"]
        self.episode_step[:, real_done] = 0.
        self.episode_return[:, real_done] = 0.
        if self.reward_type == 1:
            self.episode_return[:, cur_t==0, 1] = 0.
        self.last_action[cur_t==0] = action[0, cur_t==0]
        self.last_action[:, 1:] = action[0, :, 1:]

        if self.flags.reward_clipping > 0:
            reward = torch.clamp(reward, - self.flags.reward_clipping, self.flags.reward_clipping)

        ret = EnvOut(
            gym_env_out=gym_env_out,
            model_out=model_out,
            model_encodes=model_encodes,
            see_mask=torch.rand(size=(1, self.env_n), device=self.device) > (1 - self.actor_see_p),
            reward=reward.unsqueeze(0),
            done=done.unsqueeze(0),
            real_done=real_done.unsqueeze(0),
            truncated_done=None,
            episode_return=episode_return,
            episode_step=episode_step,
            cur_t=cur_t.unsqueeze(0),
            last_action=self.last_action.unsqueeze(0),
            max_rollout_depth=info["max_rollout_depth"].unsqueeze(0)
        )

        return (ret, model_state)

    def seed(self, seed):
        self.env.seed(seed)
    
    def close(self):
        self.env.close()

class VecModelWrapper(gym.Wrapper):
    """Wrap the gym environment with a model; output for each 
    step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out) that corresponds to underlying 
    environment frame and output from the model wrapper.
    """

    def __init__(self, env, env_n, flags, device=None, time=False):
        gym.Wrapper.__init__(self, env)
        
        self.device = torch.device("cpu") if device is None else device
        self.env = env     
        self.rec_t = flags.rec_t        
        self.flex_t = flags.flex_t 
        self.flex_t_cost = flags.flex_t_cost         
        self.discounting = flags.discounting
        self.depth_discounting = flags.depth_discounting
        self.perfect_model = flags.perfect_model
        self.tree_carry = flags.tree_carry
        self.thres_carry = flags.thres_carry        
        self.thres_discounting = flags.thres_discounting
        self.num_actions = env.action_space[0].n
        self.reward_type = flags.reward_type
        self.reward_transform = flags.reward_transform
        self.env_n = env_n
            
        if not self.flex_t:
            obs_n = 9 + self.num_actions * 10 + self.rec_t
        else:
            obs_n = 11 + self.num_actions * 10 
        
        self.observation_space = gym.spaces.Box(
          low=-np.inf, high=np.inf, shape=(env_n, obs_n, 1, 1), dtype=float)
        self.model_out_shape = (obs_n, 1, 1)
        self.gym_env_out_shape = env.observation_space.shape[1:]
        
        assert self.perfect_model, "imperfect model not yet supported"
        assert not self.thres_carry, "thres_carry not yet supported"
        assert not flags.model_rnn, "model_rnn not yet supported"

        self.baseline_max_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)
        self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)
        
        self.time = time
        if self.time: self.timings = util.Timings()
        
    def reset(self, model_net):
        """reset the environment; should only be called in the initial"""
        with torch.no_grad():
            # some init.
            self.root_max_q = [None for _ in range(self.env_n)]
            self.rollout_depth = torch.zeros(self.env_n, dtype=torch.long)
            self.max_rollout_depth = torch.zeros(self.env_n, dtype=torch.long)
            self.cur_t = torch.zeros(self.env_n, dtype=torch.long)

            # reset obs
            obs = self.env.reset()

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
            pass_action = torch.zeros(self.env_n, dtype=torch.long)
            _, vs, _, logits, encodeds = model_net(obs_py, 
                                                pass_action.unsqueeze(0).to(self.device), 
                                                one_hot=False)  
            vs = vs.cpu()
            logits = logits.cpu()

            self._debug = (obs_py, pass_action, logits, obs)
            encodeds = self.env.clone_state(inds=np.arange(self.env_n))

            # compute and update root node and current node
            self.root_nodes = []
            self.cur_nodes = []
            for n in range(self.env_n):
                root_node = Node(parent=None, action=pass_action[n].item(), logit=None, 
                num_actions=self.num_actions, discounting=self.discounting, rec_t=self.rec_t)
                encoded = {"env_state": encodeds[n], "gym_env_out": obs_py[n]}
                root_node.expand(r=torch.zeros(1, dtype=torch.float32), 
                                 v=vs[-1, n].unsqueeze(-1), 
                                 logits=logits[-1, n],
                                 encoded=encoded)
                root_node.visit()
                self.root_nodes.append(root_node)
                self.cur_nodes.append(root_node)
            
            # compute model_out
            model_out = self.compute_model_out()
            gym_env_out = torch.concat([x.encoded["gym_env_out"].unsqueeze(0) for x in self.cur_nodes])

            # record initial root_nodes_qmax 
            self.root_nodes_qmax = torch.tensor([n.max_q for n in self.root_nodes], dtype=torch.float32)
            
            return model_out.to(self.device), gym_env_out


    def step(self, action, model_net):  
        # action is tensor of shape (env_n, 3) or (env_n, 4); 
        # which corresponds to real_action, im_action, reset, term
        # all tensors in this function are in cpu, except when 
        # entering the model or outputting from the function
        
        with torch.no_grad():
            if self.time: self.timings.reset()

            action = action.cpu()
            if not self.flex_t:
                re_action, im_action, reset = action[:, 0], action[:, 1], action[:, 2]
                term = None
            else:
                re_action, im_action, reset, term = (action[:, 0], action[:, 1], 
                    action[:, 2], action[:, 3])

            # compute the mask of real / imagination step
            if not self.flex_t:
                imagine_b = self.cur_t < self.rec_t - 1
            else:
                imagine_b = (self.cur_t < self.rec_t - 1) & ~(term.bool())
            
            self.cur_t += 1
            self.cur_t[~imagine_b] = 0
            self.depth_delta = torch.pow(self.depth_discounting, self.rollout_depth)
            self.rollout_depth += 1
            self.rollout_depth[~imagine_b] = 0        
            max_rollout_depth = self.max_rollout_depth.clone()
            self.max_rollout_depth[~imagine_b] = 0
            self.max_rollout_depth = torch.max(self.max_rollout_depth, self.rollout_depth)

            # record baseline before moving on
            if torch.any(~imagine_b):                
                mean_q = [torch.mean(torch.concat(r.rollout_qs)).unsqueeze(0) for n, r in enumerate(self.root_nodes) if not imagine_b[n]]
                self.baseline_mean_q[~imagine_b] = (torch.concat(mean_q) / self.discounting).to(self.device)
                max_q = [torch.max(torch.concat(r.rollout_qs)).unsqueeze(0) for n, r in enumerate(self.root_nodes) if not imagine_b[n]]
                self.baseline_max_q[~imagine_b] = (torch.concat(max_q) / self.discounting).to(self.device)

            # four status: 
            # 1. real transition; 
            # 2. imaginary transition and expanded
            # 3. imaginary transition and done and unexpanded; 
            # 4. imagainary transition and not done and unexpanded
            # compute the status here
            
            status = torch.zeros(self.env_n, dtype=torch.long)
            status[~imagine_b] = 1             

            if torch.any(imagine_b):
                sl_cur_nodes = [x for n, x in enumerate(self.cur_nodes) if imagine_b[n]]
                sl_next_nodes = [x.children[im_action[n]] for n, x in enumerate(self.cur_nodes) if imagine_b[n]]            
                status[imagine_b] = torch.tensor([2 if y.expanded() else (3 if 'done' in x.encoded else 4) for x, y in zip(sl_cur_nodes, sl_next_nodes)],
                    dtype=torch.long)
            self._status = status
            if self.time: self.timings.time("misc_1")
            # use the model; only status 1 and 4 need to use env and model
            if torch.any((status == 1) | (status == 4)):
                sel_inds = torch.arange(self.env_n)[(status == 1) | (status == 4)]
                real_sel_b = status[(status == 1) | (status == 4)] == 1
                pass_env_states = [self.root_nodes[n].encoded["env_state"] if s == 1 else 
                    self.cur_nodes[n].encoded["env_state"] for n, s in enumerate(status) if s in [1, 4]]  
                self.env.restore_state(pass_env_states, inds=sel_inds.numpy())
                pass_action = torch.tensor([re_action[n] if s == 1 else 
                    im_action[n] for n, s in enumerate(status) if s in [1, 4]], dtype=torch.long)
                obs, reward, done, info = self.env.step(pass_action.numpy(), inds=sel_inds.numpy()) 
                real_done = [i["real_done"] if "real_done" in i else done[n] for n, i in enumerate(info)]
                if self.time: self.timings.time("step_state")

                # reset if done and the transition is real
                reset_needed = torch.zeros(self.env_n, dtype=torch.bool)
                reset_needed[sel_inds] = torch.tensor(done, dtype=torch.bool)
                reset_needed = reset_needed & (status == 1)

                if torch.any(reset_needed):                    
                    reset_inds = torch.arange(self.env_n)[reset_needed]
                    reset_m = reset_needed[sel_inds].numpy()                    
                    obs_reset = self.env.reset(inds=reset_inds.numpy()) 

                    obs[reset_m] = obs_reset
                    pass_action[reset_needed[sel_inds]] = 0    

                if self.time: self.timings.time("misc_2")
                obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
                _, vs, _, logits, encodeds = model_net(obs_py, 
                                                    pass_action.unsqueeze(0).to(self.device), 
                                                    one_hot=False)  
                vs = vs.cpu()
                logits = logits.cpu()
                if self.time: self.timings.time("model")
                encodeds = self.env.clone_state(inds=sel_inds.numpy())    
                if self.time: self.timings.time("clone_state")
            else:
                if self.time: 
                    self.timings.time("restore_state")
                    self.timings.time("misc_2")
                    self.timings.time("model")
                    self.timings.time("clone_state")

                reset_needed = torch.zeros(self.env_n, dtype=torch.bool)

            m_ind = 0
            root_nodes, cur_nodes = [], []

            # compute the current and root nodes
            for n in range(self.env_n):
                if status[n] == 1:
                    # real transition
                    new_root = (not self.tree_carry or 
                        not self.root_nodes[n].children[re_action[n]].expanded() or done[m_ind])
                    if new_root:
                        root_node = Node(parent=None, action=pass_action[m_ind].item(), logit=None, 
                            num_actions=self.num_actions, discounting=self.discounting, rec_t=self.rec_t)
                        encoded = {"env_state": encodeds[m_ind], "gym_env_out": obs_py[m_ind]}
                        root_node.expand(r=torch.zeros(1, dtype=torch.float32), 
                                        v=vs[-1, m_ind].unsqueeze(-1), 
                                        logits=logits[-1, m_ind],
                                        encoded=encoded)
                        root_node.visit()
                    else:
                        root_node = self.root_nodes[n].children[re_action[n]]
                        encoded = {"env_state": encodeds[m_ind], "gym_env_out": obs_py[m_ind]}
                        root_node.expand(r=torch.zeros(1, dtype=torch.float32), 
                                            v=vs[-1, m_ind].unsqueeze(-1), 
                                            logits=logits[-1, m_ind],
                                            encoded=encoded, 
                                            override=True)
                        root_node.parent = None
                        root_node.visit()
                    
                    root_nodes.append(root_node)
                    cur_nodes.append(root_node)
                    m_ind += 1
                
                elif status[n] == 2:
                    cur_node = self.cur_nodes[n].children[im_action[n]]
                    cur_node.visit()
                    root_nodes.append(self.root_nodes[n])
                    cur_nodes.append(cur_node)                    

                elif status[n] == 3:
                    par_logits = torch.concat([ch.logit for ch in self.cur_nodes[n].children])  
                    cur_node = self.cur_nodes[n].children[im_action[n]]
                    cur_node.expand(r=torch.zeros(1, dtype=torch.float32), 
                                    v=torch.zeros(1, dtype=torch.float32),
                                    logits=par_logits, 
                                    encoded=self.cur_nodes[n].encoded) 
                    cur_node.visit()
                    root_nodes.append(self.root_nodes[n])
                    cur_nodes.append(cur_node)
                    
                elif status[n] == 4:
                    encoded = {"env_state": encodeds[m_ind], "gym_env_out": obs_py[m_ind]}
                    if done[m_ind]: encoded["done"] = True
                    cur_node = self.cur_nodes[n].children[im_action[n]]
                    cur_node.expand(r=torch.tensor([reward[m_ind]], dtype=torch.float32), 
                                v=vs[-1, m_ind].unsqueeze(0) if not done[m_ind] else torch.zeros(1, dtype=torch.float32),
                                logits=logits[-1, m_ind], 
                                encoded=encoded) 
                    cur_node.visit()
                    root_nodes.append(self.root_nodes[n])
                    cur_nodes.append(cur_node)
                    
                    m_ind += 1

            self.root_nodes = root_nodes
            self.cur_nodes = cur_nodes
        if self.time: self.timings.time("compute_root_cur_nodes")

        # compute model_out
        model_out = self.compute_model_out(action, imagine_b, reset_needed)
        gym_env_out = torch.concat([x.encoded["gym_env_out"].unsqueeze(0) for x in self.cur_nodes])        
        if self.time: self.timings.time("compute_model_out")

        # compute reward
        if self.reward_type == 1:
            root_nodes_qmax = torch.tensor([n.max_q for n in self.root_nodes], dtype=torch.float32)        
            im_reward = torch.zeros(self.env_n,  dtype=torch.float32)        
            if torch.any(imagine_b):
                flex_t_cost = 0. if not self.flex_t else self.flex_t_cost
                im_reward[imagine_b] = ((root_nodes_qmax - self.root_nodes_qmax)*self.depth_delta - flex_t_cost)[imagine_b] # imagine reward

        re_reward = torch.zeros(self.env_n,  dtype=torch.float32)
        if torch.any(~imagine_b):            
            re_reward[~imagine_b] = torch.tensor(reward, dtype=torch.float32)[real_sel_b] # real reward            

        if self.reward_type == 1:
            full_reward = torch.concat([re_reward.unsqueeze(-1), im_reward.unsqueeze(-1)], dim=-1)
            self.root_nodes_qmax = root_nodes_qmax
        else:
            full_reward = re_reward.unsqueeze(-1)
        
        if self.time: self.timings.time("compute_reward")
        full_reward = full_reward.to(self.device)

        # compute done
        full_done = torch.zeros(self.env_n, dtype=torch.bool)
        if torch.any(~imagine_b):
            full_done[~imagine_b] = torch.tensor(done, dtype=torch.bool)[real_sel_b]
        full_done = full_done.to(self.device)

        # compute reset
        self.compute_reset(reset)

        # some info        
        full_real_done = torch.zeros(self.env_n, dtype=torch.bool)
        if torch.any(~imagine_b):
            full_real_done[~imagine_b] = torch.tensor(real_done, dtype=torch.bool)[real_sel_b]
        full_real_done = full_real_done.to(self.device)
        info = {"cur_t": self.cur_t.to(self.device),
                "max_rollout_depth": max_rollout_depth.to(self.device),
                "real_done": full_real_done}


        if self.time: self.timings.time("end")

        return (model_out.to(self.device), gym_env_out), full_reward, full_done, info

    def compute_model_out(self, action=None, imagine_b=None, reset_needed=None):
        root_nodes_stat = []
        cur_nodes_stat = []
        for n in range(self.env_n):
            root_nodes_stat.append(self.root_nodes[n].stat(detailed=True, reward_transform=self.reward_transform).unsqueeze(0))
            cur_nodes_stat.append(self.cur_nodes[n].stat(detailed=False, reward_transform=self.reward_transform).unsqueeze(0))     
        root_nodes_stat = torch.concat(root_nodes_stat)
        cur_nodes_stat = torch.concat(cur_nodes_stat)
        if action is None: 
            reset = torch.ones(self.env_n, 1, dtype=torch.float32)    
        else:
            reset = action[:, 2].unsqueeze(1)
            reset = reset.clone()
            reset[~imagine_b] = 1.
        depc = (self.discounting ** (self.rollout_depth)).unsqueeze(-1)
        if not self.flex_t:
            time = F.one_hot(self.cur_t, self.rec_t).float()
        else:
            time = (self.discounting ** (self.cur_t)).unsqueeze(-1)
        
        if not self.flex_t:
            ret_list = [root_nodes_stat, cur_nodes_stat, reset, time, depc]
        else:
            if action is None: 
                term = torch.zeros(self.env_n, 1, dtype=torch.float32)    
            else:
                term = action[:, 3].unsqueeze(1)   
                if torch.any(reset_needed):
                    term = term.clone()
                    term[reset_needed] = 0.
            ret_list = [root_nodes_stat, cur_nodes_stat, reset, term, time, depc]            
        model_out = torch.concat(ret_list, dim=-1)  
        return model_out

    def compute_reset(self, reset):
        reset_b = (reset == 1)
        self.rollout_depth[reset_b] = 0
        for n in torch.arange(self.env_n)[reset_b]:
            self.cur_nodes[n] = self.root_nodes[n]
            self.cur_nodes[n].visit()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

    def print_time(self):
        print(self.timings.summary())        

class Node:
    def __init__(self, parent, action, logit, num_actions, discounting, rec_t, device=None):        
        self.device = torch.device("cpu") if device is None else device

        self.action = F.one_hot(torch.tensor(action, dtype=torch.long), num_actions) # shape (1, num_actions)        
        self.r = torch.tensor([0.], dtype=torch.float32)    
        self.v = torch.tensor([0.], dtype=torch.float32)            
        self.logit = logit # shape (1,)        
        
        self.rollout_qs = []  # list of tensors of shape (1,)
        self.rollout_n = 0
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
        self.trail_r = torch.zeros(1, dtype=torch.float32)    
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
            
    def stat(self, detailed, reward_transform):
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

        if detailed:
            self.trail_r_undiscount = self.trail_r / self.discounting
            self.rollout_q_undiscount = self.rollout_q / self.discounting
            self.max_q = torch.max(torch.concat(self.rollout_qs) - self.r).unsqueeze(-1) / self.discounting            
        
        self.child_rollout_ns = torch.tensor([x.rollout_n for x in self.children], dtype=torch.long)
        self.child_rollout_ns_enc = self.child_rollout_ns / self.rec_t     
        ret_list = ["action", "r", "v", "child_logits", "child_rollout_qs_mean",
                    "child_rollout_qs_max", "child_rollout_ns_enc"]
        if detailed: ret_list.extend(["trail_r_undiscount", "rollout_q_undiscount", "max_q"])
        if not reward_transform:
            self.ret_dict = {x: getattr(self, x) for x in ret_list}
        else:
            tran_list = ["r", "v", "child_rollout_qs_mean", "child_rollout_qs_max", "trail_r_undiscount", "rollout_q_undiscount", "max_q"]
            self.ret_dict = {x: enc(getattr(self, x)) if x in tran_list else getattr(self, x)  for x in ret_list}
        #for x in ret_list: print(x, getattr(self, x))
        out = torch.concat(list(self.ret_dict.values()))        
        return out         

def enc(x):
    eps=0.001
    return torch.sign(x)*(torch.sqrt(torch.abs(x)+1)-1)+eps*x

# Standard wrappers

class TransposeWrap(gym.ObservationWrapper):
    """Image shape to channels x weight x height"""
    
    def __init__(self, env):
        super(TransposeWrap, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


class NoopWrapper(gym.Wrapper):
    def __init__(self, env, cost=0.):
        gym.Wrapper.__init__(self, env)
        env.action_space.n += 1    
        self.cost = cost    
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        #obs = obs[np.newaxis, :, :, :]
        self.last_obs = obs
        return obs
    
    def step(self, action):     
        if action == 0:
            return self.last_obs, self.cost, False, {}
        else:            
            obs, reward, done, info = self.env.step(action-1)
            #obs = obs[np.newaxis, :, :, :]
            self.last_obs = obs
            return obs, reward, done, info 
            
    def get_action_meanings(self):
        return ["NOOP",] + self.env.get_action_meanings()

    def clone_state(self):
        state = self.env.clone_state()
        state["noop_last_obs"] = np.copy(self.last_obs)
        return state

    def restore_state(self, state):
        self.last_obs = np.copy(state["noop_last_obs"])
        self.env.restore_state(state)
        return         

class TimeLimit_(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit_, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def clone_state(self):
        state = self.env.clone_state()
        state["timeLimit_elapsed_steps"] = self._elapsed_steps
        return state

    def restore_state(self, state):
        self._elapsed_steps = state["timeLimit_elapsed_steps"] 
        self.env.restore_state(state)
        return 

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=False, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

# Atari-related wrapped (taken from torchbeast)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        info["real_done"] = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives        
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def clone_state(self):
        state = self.env.clone_state()
        state["eps_life_vars"] = [self.lives, self.was_real_done]
        return state

    def restore_state(self, state):
        self.lives, self.was_real_done = state["eps_life_vars"] 
        self.env.restore_state(state)
        return state

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        #return np.concatenate(list(self.frames), axis=-1)
        return LazyFrames(list(self.frames))

    def clone_state(self):
        state = self.env.clone_state()
        state["frameStack"] = [np.copy(i) for i in self.frames]
        return state
    
    def restore_state(self, state):
        for i in state["frameStack"]: self.frames.append(i)
        self.env.restore_state(state)



class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class StateWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def clone_state(self):
        #state = self.env.clone_state()
        state = self.env.clone_state(include_rng=True)  
        return {"ale_state": state}
    
    def restore_state(self, state):
        #self.env.restore_state(state["ale_state"])  
        self.env.restore_state(state["ale_state"])  

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=96, height=96, grayscale=False)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

# Misc.

def align(model_out, flex_t):    
    """A function that converts v1 model_out to v0 model_out"""

    num_actions = 5
    new_model_out = torch.clone(model_out)
    s = 5 * num_actions + 2
    if flex_t:        
        new_model_out[s:2*s] = model_out[s+3:2*s+3]
        new_model_out[2*s:2*s+3] = model_out[s:s+3]     
        new_model_out[-3] = model_out[-1]  
        new_model_out[-2:] = model_out[-3:-1]  
    else:
        new_model_out[-3:] = model_out[s:s+3]
        new_model_out[s:-3] = model_out[s+3:]
    return new_model_out
