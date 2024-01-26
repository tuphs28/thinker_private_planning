import numpy as np
import gym
from gym import spaces
import torch
import torch.nn.functional as F
import thinker.util as util


class SimWrapper():
    def __init__(self, env, env_n, flags, model_net, device=None, time=False):        

        if flags.test_rec_t > 0:
            self.rec_t = flags.test_rec_t
            self.rep_rec_t = flags.rec_t         
        else:
            self.rec_t = flags.rec_t           
            self.rep_rec_t = flags.rec_t         
        self.discounting = flags.discounting
        self.max_depth = flags.max_depth        
        self.env_n = env_n
        self.env = env        
        self.device = torch.device("cpu") if device is None else device       

        action_space =  env.action_space[0]
        if type(action_space) == spaces.discrete.Discrete:    
            self.tuple_action = False                
            self.num_actions = action_space.n    
            self.dim_actions = 1
            self.dim_rep_actions = self.num_actions
            self.pri_action_shape = (self.env_n,)
        else:
            self.tuple_action = True
            self.num_actions = action_space[0].n    
            self.dim_actions = len(action_space)    
            self.dim_rep_actions = self.dim_actions
            self.pri_action_shape = (self.env_n, self.dim_actions)
        self.reset_action_shape = (self.env_n,)
        
        if env.observation_space.dtype == 'uint8':
            self.state_dtype = torch.uint8
        elif env.observation_space.dtype == 'float32':
            self.state_dtype = torch.float32
        else:
            raise Exception(f"Unupported observation sapce", env.observation_space)

        self.return_h = flags.return_h  
        self.return_x = flags.return_x
        self.im_enable = flags.im_enable        
        self.time = time        

        assert not flags.cur_enable 
        assert flags.sample_n <= 0
         
        self.timings = util.Timings()         

        self.query_size = flags.se_query_size
        self.td_lambda = flags.se_td_lambda
        self.query_cur = flags.se_query_cur # 0 for no query for current; 1 for query based on predicted z
        self.buffer_n = flags.se_buffer_n
        
        self.root_v = torch.zeros(self.env_n, device=self.device)
        self.batch_idx = torch.arange(self.env_n, device=self.device)
        self.np_batch_idx = np.arange(self.env_n)

        self.tree_rep_meaning = None
        self.obs_n = self.reset(model_net)["tree_reps"].shape[1]
        print("Tree rep shape: ", self.obs_n)
        print("Tree rep meaning: ", self.tree_rep_meaning)
        self.observation_space = {
            "tree_reps": spaces.Box(low=-np.inf, high=np.inf, shape=(self.env_n, self.obs_n), 
                dtype=np.float32),
            "real_states": self.env.observation_space,
        }        

        if self.return_x:
            xs_shape = list(self.env.observation_space.shape)
            xs_space = spaces.Box(low=0., high=1., shape=xs_shape, dtype=np.float32)
            self.observation_space["xs"] = xs_space

        if self.return_h:
            hs_shape = [env_n,] + list(model_net.hidden_shape)
            hs_space = spaces.Box(low=-np.inf, high=np.inf, shape=hs_shape, dtype=np.float32)
            self.observation_space["hs"] = hs_space

        self.observation_space = spaces.Dict(self.observation_space)        
        reset_space = spaces.Tuple((spaces.Discrete(2),)*self.env_n)
        self.action_space = spaces.Tuple((env.action_space, reset_space))


    def reset(self, model_net):
        with torch.no_grad():            
            if self.query_cur:
                self.node_key, self.node_td, self.node_action, self.node_mask = None, None, None, None

            real_state = self.env.reset()
            real_state = torch.tensor(real_state, dtype=self.state_dtype, device=self.device)
            pri_action = torch.zeros(self.env_n, self.dim_actions, dtype=torch.long, device=self.device)
            real_reward = torch.zeros(self.env_n, device=self.device)
            done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)
            per_state = model_net.initial_state(batch_size=self.env_n, device=self.device)
            model_net_out = self.real_step_model(real_state, pri_action, real_reward, done, model_net, per_state)
            
            return self.prepare_state(model_net_out, real_state)
        
    def real_step_model(self, real_state, pri_action, real_reward, done, model_net, per_state):
        if torch.any(done):
            pri_action = pri_action.clone()
            pri_action[done] = 0
        model_net_out = model_net(x=real_state, 
                                  done=None,
                                  actions=pri_action.unsqueeze(0), 
                                  state=per_state,
                                  one_hot=False,
                                  ret_xs=self.return_x,
                                  ret_zs=True,
                                  ret_hs=True)       
        self.root_per_state = model_net_out.state
        self.tmp_state = self.root_per_state

        pri_action = util.encode_action(pri_action, self.dim_actions, self.num_actions)

        self.root_td = real_reward + self.discounting * model_net_out.vs[-1] - self.root_v
        self.root_action = pri_action
        self.root_r = real_reward
        self.root_v = model_net_out.vs[-1] # (self.env_n)
        self.root_logits = model_net_out.logits[-1] # (self.env_n, self.dim_actions, self.num_actions)

        if torch.any(done):
            self.root_r = self.root_r.clone()
            self.root_td[done] = 0.
            self.root_r[done] = 0.
            if self.query_cur: self.node_mask[done, :-1] = 1 

        self.cur_td = self.root_td
        self.cur_action = self.root_action
        self.cur_r = self.root_r
        self.cur_v = self.root_v
        self.last_v = self.cur_v
        self.cur_logits = self.root_logits            

        self.cur_reset = torch.zeros(self.env_n, device=self.device)

        self.action_seq = torch.zeros(self.env_n, self.max_depth, self.dim_rep_actions, device=self.device)
        self.trail_r = torch.zeros(self.env_n, device=self.device)           
        self.trail_discount = torch.ones(self.env_n, device=self.device) 
        self.trail_lambda = torch.ones(self.env_n, device=self.device) 
        self.rollout_return = self.cur_v
        self.max_rollout_return = self.cur_v   
        self.sum_rollout_return = self.cur_v
        self.rollout_done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device) 

        self.root_action_table = torch.zeros(self.env_n, self.query_size, self.dim_rep_actions, device=self.device)
        self.root_td_table = torch.zeros(self.env_n, self.query_size, device=self.device)

        self.rollout_depth = torch.zeros(self.env_n, dtype=torch.long, device=self.device)
        self.max_rollout_depth = torch.zeros(self.env_n, dtype=torch.long, device=self.device)
        self.im_reward = torch.zeros(self.env_n, device=self.device)
        self.k = 0

        if self.query_cur:
            if self.node_key is None:
                key_shape = model_net_out.zs.shape[2:]
                self.node_key = torch.zeros((self.env_n, self.buffer_n, )+key_shape, device=self.device)
                self.node_td = torch.zeros(self.env_n, self.buffer_n, device=self.device)
                self.node_action = torch.zeros(self.env_n, self.buffer_n, self.dim_rep_actions, device=self.device)
                self.node_mask = torch.zeros(self.env_n, self.buffer_n, dtype=torch.bool, device=self.device)         
                self.node_lambda = torch.zeros(self.env_n, self.buffer_n, device=self.device)         
            self.node_key[:, -1] = model_net_out.zs[-1]
            self.root_key = model_net_out.zs[-1].clone()
            if self.query_cur == 1:
                self.node_mask[:, :-1] = 1 
                # allow tree carry for query_cur mode 2
            self.node_lambda[:, -1] = 1

            if self.query_cur == 1:
                self.topk_similarity = -torch.ones(self.env_n, self.query_size, device=self.device)         
                self.topk_td = torch.zeros(self.env_n, self.query_size, device=self.device)         
                self.topk_action = torch.zeros(self.env_n, self.query_size, self.dim_rep_actions, device=self.device)         
            else:
                self.topk_similarity, self.topk_action, self.topk_td = self.make_query(self.root_key, ignore_last=True)
                self.root_topk_similarity, self.root_topk_action, self.root_topk_td = self.topk_similarity, self.topk_action, self.topk_td 

        return model_net_out
        
    def step(self, action, model_net):  
        with torch.no_grad():
            assert type(action) is tuple and len(action) == 2, \
                "action should be a tuple of size 2, containing augmented action and reset"      
            
            pri_action, reset_action = action
            if isinstance(pri_action, np.ndarray) or isinstance(pri_action, list):
                pri_action = torch.tensor(pri_action, dtype=torch.long, device=self.device)
            if isinstance(reset_action, np.ndarray) or isinstance(reset_action, list):
                reset_action = torch.tensor(reset_action, dtype=torch.long, device=self.device)
            
            assert pri_action.shape == self.pri_action_shape, \
                f"action shape should be {self.pri_action_shape}, not {pri_action.shape}"
            assert reset_action.shape == self.reset_action_shape, \
                f"action shape should be {self.reset_action_shape}, not {reset_action.shape}"            
            
            reset_action = torch.clone(reset_action)
            
            if self.k < self.rec_t - 1:
                if not self.tuple_action: pri_action = pri_action.unsqueeze(-1)        
                # imagainary step
                self.k += 1
                self.rollout_depth += 1
                self.max_rollout_depth = torch.maximum(self.rollout_depth, self.max_rollout_depth)
                if self.max_depth > 0:
                    force_reset = self.rollout_depth >= self.max_depth
                    reset_action[force_reset] = 1
                
                initial_per_state = None
                model_net_out = model_net.forward_single(
                        state=self.tmp_state,
                        action=pri_action,                     
                        one_hot=False,
                        ret_xs=self.return_x,
                        ret_zs=True,
                        ret_hs=True)  
                self.tmp_state = model_net_out.state
                
                self.cur_action = util.encode_action(pri_action, self.dim_actions, self.num_actions)
                self.cur_r = model_net_out.rs[-1]
                self.cur_r[self.rollout_done] = 0.                  
                if torch.any(self.rollout_done):
                    self.cur_logits = torch.where(self.rollout_done.unsqueeze(-1).unsqueeze(-1), self.cur_logits, model_net_out.logits[-1])
                else:
                    self.cur_logits = model_net_out.logits[-1]
                # todo - test if this should be done after updating rollout_done
                
                self.rollout_done = torch.logical_or(self.rollout_done, model_net_out.dones[-1]) # cur_r should not be masked when just done
                self.cur_v = model_net_out.vs[-1]
                self.cur_v[self.rollout_done] = 0.
                self.cur_reset = reset_action
                self.cur_td = self.cur_r + self.discounting * self.cur_v * (1 - model_net_out.dones[-1].float()) - self.last_v
                self.last_v = self.cur_v

                self.action_seq[self.batch_idx, self.rollout_depth-1] = self.cur_action
                self.trail_r = self.trail_r + self.trail_discount * self.cur_r
                self.trail_discount = self.trail_discount * self.discounting
                self.rollout_return = self.trail_r + self.trail_discount * self.cur_v
                new_max_rollout_return = torch.maximum(self.max_rollout_return, self.rollout_return)
                self.im_reward = new_max_rollout_return - self.max_rollout_return
                self.max_rollout_return = new_max_rollout_return
                self.sum_rollout_return = self.sum_rollout_return + self.rollout_return

                self.root_td_table = torch.clone(self.root_td_table)
                self.root_action_table = torch.clone(self.root_action_table)

                self.root_td_table[:, 0] = self.root_td_table[:, 0] + self.trail_lambda * self.cur_td
                self.trail_lambda = self.trail_lambda * self.td_lambda * self.discounting
                one_depth_mask = self.rollout_depth == 1
                self.root_action_table[one_depth_mask, 0] = self.cur_action[one_depth_mask]

                if self.query_cur:
                    # fill the previous node
                    self.node_action[:, -1] = self.cur_action
                    self.node_td = self.node_td + self.node_lambda * self.cur_td.unsqueeze(-1)
                    self.node_lambda = self.node_lambda * self.td_lambda * self.discounting

                    query = model_net_out.zs[-1]
                    # let do a search using key
                    self.topk_similarity, self.topk_action, self.topk_td = self.make_query(query)                    

                    if self.query_cur == 2:
                        self.root_topk_similarity, self.root_topk_action, self.root_topk_td = self.make_query(self.root_key)                    


                state = self.prepare_state(model_net_out, None)
                # reset processing
                reset_bool = reset_action.bool()
                if torch.any(reset_bool):
                    self.rollout_depth[reset_bool] = 0
                    self.action_seq[reset_bool] = 0
                    self.trail_r[reset_bool] = 0
                    self.trail_discount[reset_bool] = 1
                    self.trail_lambda[reset_bool] = 1
                    self.root_td_table[reset_bool, 1:] = self.root_td_table[reset_bool, :-1]
                    self.root_td_table[reset_bool, 0] = 0
                    self.root_action_table[reset_bool, 1:] = self.root_action_table[reset_bool, :-1]
                    self.rollout_done[reset_bool] = 0
                    self.last_v[reset_bool] = self.root_v[reset_bool]
                    for k in self.tmp_state.keys():
                        self.tmp_state[k][reset_bool] = self.root_per_state[k][reset_bool]

                    self.im_reward[reset_bool] = 0
                    # todo - test if gating planning reward is necessary

                if self.query_cur:                    
                    # shift table
                    self.node_action[:, :-1] = self.node_action[:, 1:].clone()
                    self.node_action[:, -1] = 0 # to be filled on the next step
                    self.node_td[:, :-1] = self.node_td[:, 1:].clone()
                    self.node_td[:, -1] = 0 # to be filled on the next step
                    self.node_mask[:, :-1] = self.node_mask[:, 1:].clone()
                    self.node_mask[:, -1] = 0 
                    self.node_lambda[:, :-1] = self.node_lambda[:, 1:].clone()
                    self.node_lambda[:, -1] = 1
                    if torch.any(reset_bool):
                        self.node_lambda[reset_bool, :-1] = 0
                    self.node_key[:, :-1] = self.node_key[:, 1:].clone()
                    self.node_key[:, -1] = query
                    if torch.any(reset_bool):
                        self.node_key[reset_bool, -1] = self.root_key[reset_bool]

                real_reward =  torch.zeros(self.env_n, device=self.device)
                done = torch.zeros(self.env_n, device=self.device, dtype=torch.bool)
                truncated_done = done
                real_done = done

                if self.k < self.rec_t - 1:
                    step_status = torch.ones(self.env_n, dtype=torch.long, device=self.device)
                else:
                    step_status = torch.ones(self.env_n, dtype=torch.long, device=self.device) * 2            
                baseline = None
            else:
                # real step
                baseline = self.sum_rollout_return / self.rec_t 
                real_state, real_reward, done, info = self.env.step(pri_action.detach().cpu().numpy())
                if np.any(done): 
                    new_real_state = self.env.reset(self.np_batch_idx[done])
                    real_state[done] = new_real_state
                real_state = torch.tensor(real_state, dtype=self.state_dtype, device=self.device)
                real_reward = torch.tensor(real_reward, dtype=torch.float, device=self.device)
                done = torch.tensor(done, dtype=torch.bool, device=self.device)

                initial_per_state = self.root_per_state
                if not self.tuple_action: pri_action = pri_action.unsqueeze(-1)        
                model_net_out = self.real_step_model(real_state, pri_action, real_reward, done, model_net, initial_per_state)
                state = self.prepare_state(model_net_out, real_state)
                if "real_done" in info[0].keys():
                    real_done = [m["real_done"] for m in info]
                    real_done = torch.tensor(real_done, dtype=torch.bool, device=self.device)
                else:
                    real_done = done

                if "truncated_done" in info[0].keys():
                    truncated_done = [m["truncated_done"] for m in info]
                    truncated_done = torch.tensor(truncated_done, dtype=torch.bool, device=self.device)
                else:
                    truncated_done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)

                step_status = torch.zeros(self.env_n, dtype=torch.long, device=self.device)

            info = {
                "real_done": real_done,
                "truncated_done": truncated_done,
                "step_status": step_status,
                "max_rollout_depth": self.max_rollout_depth,
                "baseline": baseline,
                "initial_per_state": initial_per_state,
                "im_reward": self.im_reward,
            }

            return state, real_reward, done, info
        
    def prepare_state(self, model_net_out, real_state):
        self.one_hot_k = torch.zeros(self.env_n, self.rec_t, device=self.device)
        self.one_hot_k[:, self.k] = 1
        root_stat = ["root_td", "root_action", "root_r", "root_v", "root_logits"]
        cur_stat = ["cur_td", "cur_action", "cur_r", "cur_v", "cur_logits"]
        misc = ["cur_reset", "one_hot_k", "rollout_return", "max_rollout_return", "rollout_done"]
        root_tables = ["root_action_table", "root_td_table"]
        action_seq = ["action_seq"]   
        cur_topk = ["topk_similarity", "topk_td", "topk_action"]
        root_topk = ["root_topk_similarity", "root_topk_td", "root_topk_action"]

        tree_reps = []
        tree_rep_keys = root_stat + cur_stat + misc + action_seq
        if self.query_cur == 1: 
            tree_rep_keys += root_tables + cur_topk
        elif self.query_cur == 2: 
            tree_rep_keys += root_topk + cur_topk
        else:
            tree_rep_keys += root_tables 
        for k in tree_rep_keys:
            v = getattr(self, k)
            if len(v.shape) == 1:
                v = v.unsqueeze(-1)
            elif len(v.shape) > 2:
                v = torch.flatten(v, start_dim=1)
            tree_reps.append(v.float())

        if self.tree_rep_meaning is None:
            self.tree_rep_meaning = {}
            idx = 0
            for n, k in enumerate(tree_rep_keys):
                next_idx = idx + tree_reps[n].shape[1]
                self.tree_rep_meaning[k] = slice(idx, next_idx)
                idx = next_idx            
        
        tree_reps = torch.concat(tree_reps, dim=1)
        state = {
            "tree_reps": tree_reps,
            "real_states": real_state,
        }
        if self.return_x:
            state["xs"] = model_net_out.xs[-1]
        if self.return_h:
            state["hs"] = model_net_out.hs[-1]
        return state
    
    def make_query(self, query, ignore_last=False):
        expanded_query = query.unsqueeze(1).expand_as(self.node_key)
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            torch.flatten(self.node_key, 2), 
            torch.flatten(expanded_query, 2), 
            dim=-1)  # size (env_n, buffer_n)
        similarity[self.node_mask] = -1. # previous episode / stage similarity is all set to minimum
        if ignore_last: similarity[:, -1] = -1
        topk_similarity, topk_indicies = torch.topk(similarity, k=self.query_size, dim=-1)
        # get top k indicies and collect respective action and td
        topk_action = torch.gather(self.node_action, dim=1, index=topk_indicies.unsqueeze(-1).expand(-1, -1, self.dim_rep_actions))
        topk_td = torch.gather(self.node_td, dim=1, index=topk_indicies)
        # set masked encoding to 0
        topk_action[topk_similarity==-1] = 0
        topk_td[topk_similarity==-1] = 0
        return topk_similarity, topk_action, topk_td

    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)
    
    def close(self):
        self.env.close()

    def seed(self, x):
        self.env.seed(x)
    
    def clone_state(self, inds=None):
        return self.env.clone_state(inds)

    def restore_state(self, state, inds=None):
        self.env.restore_state(state, inds)

    def unwrapped_step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    def get_action_meanings(self):
        return self.env.get_action_meanings()   
    
    def decode_tree_reps(self, tree_reps):
        if len(tree_reps.shape) == 3: tree_reps = tree_reps[0]
        return {k: tree_reps[:, v] for k, v in self.tree_rep_meaning.items()}
