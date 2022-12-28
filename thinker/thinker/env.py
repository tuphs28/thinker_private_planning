from collections import namedtuple
import numpy as np
import torch
from torch.nn import functional as F
import gym
import gym_csokoban

EnvOut = namedtuple('EnvOut', ['gym_env_out', 'model_out', 'reward', 'done', 
    'truncated_done', 'episode_return', 'episode_step', 'cur_t', 'last_action',
    'max_rollout_depth'])

def Environment(flags, model_wrap=True):
    """Create an environment using flags.env; first
    wrap the env with basic wrapper, then wrap it
    with a model, and finally wrap it with PosWrapper
    for episode processing

    Args:
        flags (namespace): configuration flags
        model_wrap (boolean): whether to wrap the environment with model
    Return:
        environment
    """
    if model_wrap:
        return PostWrapper(ModelWrapper(PreWrap(gym.make(flags.env), flags.env), flags), True, flags)
    else:
        return PostWrapper(PreWrap(gym.make(flags.env), flags.env), False, flags)

def PreWrap(env, name):
    if name == "cSokoban-v0":
        env = TransposeWrap(env)
    else:
        raise ValueError("Environment not supported")

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

        if self.model_wrap:
            out, model_state = self.env.reset(model_net)
            gym_env_out, model_out = _format(*out)
        else:
            gym_env_out, model_out = _format(self.env.reset(), None)

        ret = EnvOut(
            gym_env_out=gym_env_out,
            model_out=model_out,
            reward=initial_reward,
            done=initial_done,
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
            out, reward, done, unused_info, model_state = self.env.step(action[0,0].cpu().detach().numpy(), model_net, model_state)     
            gym_env_out, model_out = _format(*out)
        else:
            out, reward, done, unused_info = self.env.step(action[0,0,0].cpu().detach().numpy())     
            gym_env_out, model_out = _format(out, None)
        self.episode_step += 1
        self.episode_return = self.episode_return + torch.tensor(reward).unsqueeze(0).unsqueeze(0)
        episode_step = self.episode_step
        episode_return = self.episode_return.clone()
        if done:
            if self.model_wrap:
                out = self.env.reset(model_net)
            else:
                out = self.env.reset()
            self.episode_return = torch.zeros(1, 1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)        
        
        reward = torch.tensor(reward).view(1, 1, -1)
        done = torch.tensor(done).view(1, 1)
        truncated_done = 'TimeLimit.truncated' in unused_info and unused_info['TimeLimit.truncated']
        truncated_done = torch.tensor(truncated_done).view(1, 1)
        if self.model_wrap:
            cur_t = torch.tensor(unused_info["cur_t"]).view(1, 1)
            if cur_t == 0 and self.episode_return.shape[2] > 1:
                self.episode_return[:, :, 1] = 0.
            if 'max_rollout_depth' in unused_info:
                max_rollout_depth = torch.tensor(unused_info["max_rollout_depth"]).view(1, 1)
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
        ret = EnvOut(
            gym_env_out=gym_env_out,
            model_out=model_out,
            reward=reward,
            done=done,
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
        self.reward_type = flags.reward_type    
        self.no_mem = flags.no_mem
        self.perfect_model = flags.perfect_model
        self.reset_m = flags.reset_m
        self.tree_carry = flags.tree_carry
        self.thres_carry = flags.thres_carry        
        self.thres_discounting = flags.thres_discounting
        self.num_actions = env.action_space.n
        self.cur_node = None
        self.root_node = None
        self.debug = False
            
        if not self.flex_t:
            obs_n = 9 + self.num_actions * 10 + self.rec_t
        else:
            obs_n = 10 + self.num_actions * 11 
        
        self.observation_space = gym.spaces.Box(
          low=-np.inf, high=np.inf, shape=(obs_n, 1, 1), dtype=float)
        
        self.max_rollout_depth = 0.
        self.thres = None
        self.root_max_q = None
        
    def reset(self, model_net, **kwargs):
        x = self.env.reset(**kwargs)
        self.cur_t = 0            
        model_state = model_net.core.init_state(1) if model_net.rnn else None        
        out, model_state = self.use_model(model_net=model_net, 
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
          if self.debug and self.cur_t == 1: self.debug_xs = []  

          self.cur_t += 1
          out, model_state = self.use_model(model_net=model_net, 
            model_state=None, x=None, r=None, a=im_action, 
            cur_t=self.cur_t, reset=reset, term=term, done=False)          
          if self.reward_type == 0:
            r = np.array([0.])
          else:
            if self.flex_t:
                flex_t_cost = self.flex_t_cost
            else:                
                flex_t_cost = 0.
            r = np.array([0., (self.root_max_q - self.last_root_max_q - flex_t_cost).item()], dtype=np.float32)
          done = False
          info['cur_t'] = self.cur_t   
          x = None
        else:
          self.cur_t = 0
          if self.perfect_model: self.env.restore_state(self.root_node.encoded["env_state"])
          x, r, done, info_ = self.env.step(re_action)                    
          out, model_state = self.use_model(model_net=model_net, 
            model_state=model_state, x=x, r=r, a=re_action, 
            cur_t=self.cur_t, reset=1., term=term, done=done) 
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
                # a_tensor is in shape (num_action,)
                if not model_net.rnn:
                    _, vs, logits, encodeds = model_net(x_tensor, a_tensor.unsqueeze(0), one_hot=True)                     
                else:
                    vs, logits, model_state = model_net(x=x_tensor.unsqueeze(0), 
                              actions=a_tensor.unsqueeze(0).unsqueeze(0), 
                              done=torch.tensor(done, dtype=bool).unsqueeze(0).unsqueeze(0),
                              state=model_state,
                              one_hot=True) 
                
                if self.perfect_model: 
                    encoded = {"env_state": self.clone_state()}
                else:
                    encoded=encodeds[-1]
                if self.debug: encoded["x"] = x
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
                if self.debug and self.cur_node is not None: self.debug_xs.append(self.cur_node.encoded["x"])
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
                            if self.debug: encoded["x"] = x
                            if done: encoded["done"] = True                        
                            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                            self.x_ = x_tensor
                            a_tensor = F.one_hot(torch.tensor(a, dtype=torch.long).unsqueeze(0), self.num_actions) 

                            if not model_net.rnn:
                                _, vs, logits, encodeds = model_net(x_tensor, a_tensor.unsqueeze(0), one_hot=True)                     
                            else:
                                vs, logits, model_state = model_net(x=x_tensor.unsqueeze(0), 
                                        actions=a_tensor.unsqueeze(0).unsqueeze(0), 
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
            
            root_node_stat = self.root_node.stat()
            cur_node_stat = self.cur_node.stat()                        
            reset = torch.tensor([reset], dtype=torch.float32)            
            depc = torch.tensor([self.discounting ** (self.rollout_depth-1)])
            
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
                if cur_t > 0 and self.debug: self.debug_xs.append(self.cur_node.encoded["x"])
                self.rollout_depth = 0
                self.unexpand_rollout_depth = 0.
                self.cur_node = self.root_node
                self.cur_node.visit()
                self.pass_unexpand = False
            
            return out, self.root_node.encoded["model_state"] if model_net.rnn else None
                
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
            
    def stat(self):
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
        self.child_rollout_ns_enc = self.child_rollout_ns / self.rec_t       
            
        ret_list = ["action", "r", "v", "child_logits", "child_rollout_qs_mean",
                    "child_rollout_qs_max", "child_rollout_ns_enc"]
        self.ret_dict = {x: getattr(self, x) for x in ret_list}
        out = torch.concat(list(self.ret_dict.values()))        
        return out                

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
