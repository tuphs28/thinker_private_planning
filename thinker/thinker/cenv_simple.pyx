# distutils: language = c++
import numpy as np
import gym
import torch
import thinker.util as util
from thinker.net import ModelNetOut

import cython
from libcpp cimport bool
from libcpp.vector cimport vector
from cpython.ref cimport PyObject

cdef class cVecSimModelWrapper():
    """Wrap the gym environment with a model; output for each 
    step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out, model_encodes) that corresponds to underlying 
    environment frame, output from the model wrapper, and encoding from the model
    Assume a learned dynamic model. Use simplified representation.
    """
    # setting
    cdef int rec_t
    cdef float discounting
    cdef float depth_discounting
    cdef int max_allow_depth 
    cdef int num_actions
    cdef int obs_n    
    cdef int env_n
    cdef bool time 
    cdef bool flatten
    cdef bool debug

    # python object
    cdef object device
    cdef object env
    cdef object timings    
    cdef readonly baseline_max_q
    cdef readonly baseline_mean_q    
    cdef readonly object model_out_shape
    cdef readonly object gym_env_out_shape
    cdef readonly object xs
    cdef int mode

    # internal variables only used in step function
    cdef int[:] cur_t
    cdef int[:] rollout_depth
    cdef float[:,:,:,:] rollout_rep
    cdef int[:] rr_row
    cdef int[:] rr_col
    cdef float[:] trail_r # trailing reward
    cdef float[:] trail_discount # trailing discount
    
    cdef readonly object root_state
    cdef readonly object cur_state

    cdef int rr_max_row
    cdef int rr_max_col
    cdef int rr_max_entry_n

    cdef int[:] max_rollout_depth
    cdef int[:] max_rollout_depth_

    

    def __init__(self, env, env_n, flags, device=None, time=False, debug=False):
        assert not flags.perfect_model, "this class only supports imperfect model"
        assert flags.im_cost <= 0, "does not support planning rewards"

        self.device = torch.device("cpu") if device is None else device
        self.env = env     
        self.rec_t = flags.rec_t                  
        self.discounting = flags.discounting
        self.max_allow_depth = flags.max_depth
        self.num_actions = env.action_space[0].n 
        self.env_n = env_n      
        self.flatten = not flags.actor_sim  
        self.mode = flags.sim_mode
        self.gym_env_out_shape = env.observation_space.shape[1:]        

        self.baseline_max_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)
        self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)        
        self.time = time        
        self.timings = util.Timings()
        self.debug = debug

        # internal variable init.        
        self.rr_max_row = max(2 * (self.rec_t // self.max_allow_depth), 4)
        self.rr_max_col = self.max_allow_depth + 1
        self.rr_max_entry_n = 2 * self.num_actions + 4
        
        if self.mode == 0:
            self.model_out_shape = (self.rr_max_row, self.rr_max_col, self.rr_max_entry_n)            
        elif self.mode == 1:
            self.model_out_shape = (1, 2, self.rr_max_entry_n)            
        elif self.mode == 2:
            self.model_out_shape = (self.rr_max_row+1, self.rr_max_col, self.rr_max_entry_n)            
        self.obs_n = int(np.prod(self.model_out_shape))
        if self.flatten:            
            self.model_out_shape = (self.obs_n, 1, 1)            

    def reset(self, model_net):
        """reset the environment; should only be called in the initial"""  
        cdef int[:] action
        cdef float[:] rs

        self.rr_row = np.zeros(self.env_n, dtype=np.intc)
        self.rr_col = np.zeros(self.env_n, dtype=np.intc)
        self.rollout_rep = np.zeros((self.env_n, self.rr_max_row, self.rr_max_col, self.rr_max_entry_n), dtype=np.float32)
        self.trail_r = np.zeros(self.env_n, dtype=np.float32)
        self.trail_discount = np.ones(self.env_n, dtype=np.float32)     

        self.max_rollout_depth = np.zeros(self.env_n, dtype=np.intc)
        self.max_rollout_depth_ = np.zeros(self.env_n, dtype=np.intc)

        self.cur_t = np.zeros(self.env_n, dtype=np.intc)
        self.rollout_depth = np.zeros(self.env_n, dtype=np.intc)

        with torch.no_grad():
             # reset obs
            obs = self.env.reset()

            # obtain output from model            
            obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
            re_action = np.zeros(self.env_n, dtype=np.intc)
            re_action_py = torch.tensor(re_action, dtype=torch.long).unsqueeze(0).to(self.device)
            model_net_out = model_net(obs_py, re_action_py, one_hot=False)  

            rs = np.zeros(self.env_n, dtype=np.float32)
            vs = model_net_out.vs[0].cpu().numpy()     
            dones = np.zeros(self.env_n, dtype=np.float32)
            logits = model_net_out.logits[0].cpu().numpy()    
           
            self.update_rr(re_action, rs, vs, dones, logits)
            ys = model_net_out.ys[0]
            self.root_state = model_net_out.state  
            self.cur_state = self.root_state          

            rep = self.compute_rep()
            rollout_rep_py = torch.tensor(rep, dtype=torch.float32, device=self.device)            
            if self.flatten: rollout_rep_py = torch.flatten(rollout_rep_py, start_dim=1)      

        if self.debug: self.xs = self.cur_state["pred_xs"]

        return rollout_rep_py, obs_py, ys

    def step(self, action, model_net):  
        cdef int i, j
        cdef int[:] re_action_
        cdef int[:] im_action_
        cdef int[:] reset
        cdef vector[int] re_idx
        cdef vector[int] im_idx
        cdef vector[int] re_action        
        cdef vector[int] im_action
        cdef vector[int] env_reset_idx
        cdef vector[int] env_reset_inner_idx
        
        cdef int[:] update_action
        cdef float[:] rs
        cdef float[:] vs
        cdef float[:,:] logits
        cdef bool cloned
        
        action = action.cpu().int().numpy()
        re_action_, im_action_, reset = action[:, 0], action[:, 1], action[:, 2]
        for i in range(self.env_n):  
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
            if self.cur_t[i] < self.rec_t - 1: # imagaination step
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(self.max_rollout_depth[i], self.rollout_depth[i])
                self.trail_discount[i] *= self.discounting
                im_idx.push_back(i)
                im_action.push_back(im_action_[i])    
            else:   # real step
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0   
                self.max_rollout_depth[i] = 0
                re_idx.push_back(i)
                re_action.push_back(re_action_[i]) 
                bl_sum = np.sum(self.rollout_rep[i, :, 1:, self.num_actions+2]) + self.rollout_rep[i, 0, 0, self.num_actions+2]
                bl_count = np.sum(self.rollout_rep[i, :, 1:, :self.num_actions]) + 1
                self.baseline_mean_q[i] = bl_sum / bl_count

        self.clear_rr(re_idx)
        if self.time: self.timings.time("misc_1")

        # one step of env
        if re_idx.size() > 0:
            obs, env_reward, env_done, env_info = self.env.step(re_action, inds=re_idx) 
            env_real_done = [m["real_done"] if "real_done" in m else env_done[n] for n, m in enumerate(env_info)]
            env_truncated_done = [m["truncated_done"] if "truncated_done" in m else False for n, m in enumerate(env_info)]
 
        if self.time: self.timings.time("step_state")

        # env reset needed?
        for i, j in enumerate(re_idx):
            if env_done[i]:
                env_reset_idx.push_back(j)
                env_reset_inner_idx.push_back(i) # index within pass_inds_step        
  
        # env reset
        if env_reset_idx.size() > 0:
            obs_reset = self.env.reset(inds=env_reset_idx) 
            for i, j in enumerate(env_reset_inner_idx):
                obs[j] = obs_reset[i]
                re_action[j] = 0   
        if self.time: self.timings.time("step_state")

        # use model for real transition
        if re_idx.size() > 0:
            with torch.no_grad():
                obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
                re_action_py = torch.tensor(re_action, dtype=long, device=self.device).unsqueeze(0)
                re_model_net_out = model_net(obs_py, re_action_py, one_hot=False)
                re_vs = re_model_net_out.vs[-1].float().cpu().numpy()                
                re_logits = re_model_net_out.logits[-1].float().cpu().numpy()
                re_state = re_model_net_out.state
        if self.time: self.timings.time("compute_model_out_re")
   
        # use model for imaginary transition
        if im_idx.size() > 0:
            if int(im_idx.size()) == self.env_n:
                state = self.cur_state
            else:
                state = {k: v[im_idx] for k, v in self.cur_state.items()}
            with torch.no_grad():
                im_action_py = torch.tensor(im_action, dtype=long, device=self.device)
                im_model_net_out = model_net.forward_single(state, im_action_py, one_hot=False)
                im_rs = im_model_net_out.rs[-1].float().cpu().numpy()
                im_vs = im_model_net_out.vs[-1].float().cpu().numpy()
                im_dones = im_model_net_out.dones[-1].float().cpu().numpy()
                im_logits = im_model_net_out.logits[-1].float().cpu().numpy()
                im_state = im_model_net_out.state
        if self.time: self.timings.time("compute_model_out_im")
  
        # compute action, rs, vs, dones, logits for update_rr
        # three cases: all are real transition; all are imaginary transition
        # mixed transition        

        if int(re_idx.size()) == self.env_n:
            update_action = np.array(re_action, dtype=np.intc)
            rs = np.zeros(self.env_n, dtype=np.float32)
            vs = re_vs
            dones = np.zeros(self.env_n, dtype=np.float32)
            logits = re_logits
        elif int(im_idx.size()) == self.env_n:
            update_action = np.array(im_action, dtype=np.intc)
            rs = im_rs
            vs = im_vs
            dones = im_dones
            logits = im_logits
        else:
            raise Exception("Mixed transitions not yet implemented.")
        self.update_rr(update_action, rs, vs, dones, logits)

        # update root_state, cur_state, ys, reward
        if int(re_idx.size()) == self.env_n:
   
            self.root_state = re_state
            self.cur_state = re_state
            ys = re_model_net_out.ys[0]

            full_reward = torch.tensor(env_reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
            full_done = torch.tensor(env_done, dtype=torch.float32, device=self.device).bool()
            full_real_done = torch.tensor(env_real_done, dtype=torch.float32, device=self.device).bool()
            full_truncated_done = torch.tensor(env_truncated_done, dtype=torch.float32, device=self.device).bool()

        elif int(im_idx.size()) == self.env_n:
            self.cur_state = im_state
            ys = im_model_net_out.ys[0]
            obs_py = None
            full_reward = torch.zeros(self.env_n, 1, dtype=torch.float, device=self.device)
            full_done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)
            full_real_done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)
            full_truncated_done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)            
        else:
            raise Exception("Mixed transitions not yet implemented.")        

        if self.debug: self.xs = self.cur_state["pred_xs"]

        # reset if reach max depth
        if self.max_allow_depth > 0:
            for i in range(self.env_n):
                if self.rollout_depth[i] >= self.max_allow_depth:
                    reset[i] = 1   
  
        # reset if done
        for j, i in enumerate(im_idx):
            if im_dones[j] > 0.5:
                reset[i] = 1

        if self.mode == 1: rep = self.compute_rep()

        # reset processing
        cloned = False
        for i in range(self.env_n):
            if self.cur_t[i] > 0 and reset[i]: # imagaination step
                self.rollout_depth[i] = 0  
                self.new_row_rr(i)
                if not cloned:
                    self.cur_state = {k:v.clone() for (k, v) in self.cur_state.items()}
                    cloned = True
                for k in self.cur_state.keys():                    
                    self.cur_state[k][i] = self.root_state[k][i]
            
        if self.mode in [0, 2]: rep = self.compute_rep()
        rollout_rep_py = torch.tensor(rep, dtype=torch.float32, device=self.device)            
        if self.flatten: rollout_rep_py = torch.flatten(rollout_rep_py, start_dim=1)

        # some extra info
        info = {"cur_t": torch.tensor(self.cur_t, dtype=torch.long, device=self.device),
                "max_rollout_depth":  torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
                "real_done": full_real_done,
                "truncated_done": full_truncated_done,}

        return ((rollout_rep_py, obs_py, ys), 
                full_reward,
                full_done,
                info)

    def close(self):
        self.env.close()

    def seed(self, x):
        self.env.seed(x)

    def print_time(self):
        print(self.timings.summary())

    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        self.env.restore_state(state)

    def get_action_meanings(self):
        return self.env.get_action_meanings()    

    cdef update_rr(self, int[:]& action, float[:]& rs, float[:]& vs, float[:]& dones, float[:,:]& logits):
        cdef int i
        cdef float[:] cur_entry
        for i in range(self.env_n):
            self.trail_r[i] += self.trail_discount[i] * rs[i] / self.discounting
            assert self.rr_row[i] < self.rr_max_row, "self.rr_row[i] overflows %d / %d"  % (self.rr_row[i], self.rr_max_row)
            assert self.rr_col[i] < self.rr_max_col, "self.rr_col[i] overflows %d / %d"  % (self.rr_col[i], self.rr_max_col)
            cur_entry = self.rollout_rep[i, self.rr_row[i], self.rr_col[i], :]
            cur_entry[action[i]] = 1.
            cur_entry[self.num_actions] = rs[i]
            cur_entry[self.num_actions+1] = vs[i]
            cur_entry[self.num_actions+2] = self.trail_r[i] + self.trail_discount[i] * vs[i]
            cur_entry[self.num_actions+3] = dones[i]
            cur_entry[self.num_actions+4:] = logits[i]
            self.rr_col[i] += 1                
    
    cdef clear_rr(self, vector[int]& idx):
        cdef int i
        for i in idx:
            self.rr_row[i] = 0
            self.rr_col[i] = 0
            self.rollout_rep[i] = 0.
            self.trail_r[i] = 0.
            self.trail_discount[i] = 1

    cdef new_row_rr(self, int i):
        if self.rr_row[i] + 1 >= self.rr_max_row:
            self.rollout_rep[i, :-1] = self.rollout_rep[i, 1:]
        else:
            self.rr_row[i] += 1
        self.rr_col[i] = 1
        self.rollout_rep[i, self.rr_row[i], 0] = self.rollout_rep[i, self.rr_row[i] - 1, 0] # replicate last row root node
        self.trail_r[i] = 0.
        self.trail_discount[i] = 1       

    cdef float[:, :, :, :] compute_rep(self):
        cdef int i, row, cur_row, col, cur_col
        cdef bool filled
        cdef float[:, :, :, :] rep
        if self.mode == 0:
            rep = self.rollout_rep
        elif self.mode == 1:
            rep_np = np.zeros((self.env_n, 1, 2, self.rr_max_entry_n), dtype=np.float32)
            rep = rep_np      
            for i in range(self.env_n):                
                rep[i, 0, 0] = self.rollout_rep[i, 0, 0] 
                rep[i, 0, 1] = self.rollout_rep[i, self.rr_row[i], self.rr_col[i]-1] 
        elif self.mode == 2:
            rep_np = np.zeros((self.env_n, self.rr_max_row+1, self.rr_max_col, self.rr_max_entry_n), dtype=np.float32)
            rep = rep_np      
            for i in range(self.env_n):        
                rep[i, 0, 0] = self.rollout_rep[i, 0, 0] 
                cur_row, cur_col = 1, 0
                for row in range(self.rr_max_row-1, -1, -1):
                    filled = False
                    for col in range(self.rr_max_col-1, -1, -1):
                        entry = self.rollout_rep[i, row, col]                            
                        if np.sum(entry[:self.num_actions]) > 0: # filled
                            filled = True
                            rep[i, cur_row, cur_col] = self.rollout_rep[i, row, col]  
                            cur_col += 1
                    if filled: 
                        cur_col = 0
                        cur_row += 1
        return rep