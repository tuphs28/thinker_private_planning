# distutils: language = c++
import numpy as np
import gym
import torch
from thinker import util

import cython
from libcpp cimport bool
from libcpp.vector cimport vector
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free

# util function

@cython.cdivision(True)
cdef float average(vector[float]& arr):
    cdef int n = arr.size()    
    if n == 0: return 0.
    cdef float sum = 0
    cdef int i    
    for i in range(n): sum += arr[i]
    return sum / n    

cdef float maximum(vector[float]& arr):
    cdef int n = arr.size()    
    if n == 0: return 0.
    cdef float max_val = arr[0]
    cdef int i
    for i in range(1, n): 
        if arr[i] > max_val: 
            max_val = arr[i]
    return max_val       

# Node-related function (we use structure instead of class to minimize Python code)

cdef struct Node:
    int action # action
    float r # reward
    float v # value
    bool done # whether done or not
    float logit # logit    
    vector[Node*]* ppchildren # children node list
    Node* pparent # parent node
    float trail_r # trailing reward
    float trail_discount # trailing discount
    float rollout_q # trailing rollout q    
    bool visited # visited?
    vector[float]* prollout_qs # all rollout return
    int rollout_n # number of rollout
    float max_q # maximum of all v
    int num_actions # number of actions
    float discounting # discount rate
    int rec_t # number of planning step
    PyObject* encoded # all python object

cdef Node* node_new(Node* pparent, int action, float logit, int num_actions, float discounting, int rec_t):
    cdef Node* pnode = <Node*> malloc(sizeof(Node))
    cdef vector[Node*]* ppchildren =  new vector[Node*]()
    cdef vector[float]* prollout_qs = new vector[float]()
    pnode[0] = Node(action=action, r=0., v=0., done=False, logit=logit, ppchildren=ppchildren, pparent=pparent, trail_r=0., trail_discount=1., rollout_q=0,
        visited=False, prollout_qs=prollout_qs, rollout_n=0, max_q=0., num_actions=num_actions, discounting=discounting, rec_t=rec_t, encoded=NULL)
    return pnode

cdef bool node_expanded(Node* pnode):
    return pnode[0].ppchildren[0].size() > 0

cdef node_expand(Node* pnode, float r, float v, bool done, float[:] logits, PyObject* encoded,bool override):
    """
    First time arriving a node and so we expand it
    """
    cdef int a    
    cdef Node* pnode_

    if not override: assert not node_expanded(pnode), "node should not be expanded"
    if override:        
        pnode[0].prollout_qs[0][0] = v * pnode[0].discounting
        for a in range(1, int(pnode[0].prollout_qs[0].size())):
            pnode[0].prollout_qs[0][a] = pnode[0].prollout_qs[0][a] - pnode[0].r + r
    pnode[0].r = r
    pnode[0].v = v
    if pnode[0].encoded != NULL: 
        Py_DECREF(<object>pnode[0].encoded)    
    pnode[0].encoded = encoded
    pnode[0].done = done
    Py_INCREF(<object>encoded)
    for a in range(pnode[0].num_actions):
        if not override:
            pnode[0].ppchildren[0].push_back(node_new(pparent=pnode, action=a, logit=logits[a], 
                num_actions = pnode[0].num_actions, discounting = pnode[0].discounting, rec_t = pnode[0].rec_t))
        else:
            pnode[0].ppchildren[0][a][0].logit = logits[a]                

cdef node_visit(Node* pnode):
    pnode[0].trail_r = 0.
    pnode[0].trail_discount = 1.
    node_propagate(pnode=pnode, r=pnode[0].r, v=pnode[0].v, new_rollout=not pnode[0].visited)
    pnode[0].visited = True

cdef void node_propagate(Node* pnode, float r, float v, bool new_rollout):
    pnode[0].trail_r = pnode[0].trail_r + pnode[0].trail_discount * r
    pnode[0].trail_discount = pnode[0].trail_discount * pnode[0].discounting
    pnode[0].rollout_q = pnode[0].trail_r + pnode[0].trail_discount * v
    if new_rollout:
        pnode[0].prollout_qs[0].push_back(pnode[0].rollout_q)
        pnode[0].rollout_n = pnode[0].rollout_n + 1        
    if pnode[0].pparent != NULL: 
        node_propagate(pnode[0].pparent, r, v, new_rollout)

#@cython.cdivision(True)
cdef float[:] node_stat(Node* pnode, bool detailed):
    cdef float[:] result = np.zeros((pnode[0].num_actions*5+5) if detailed else (pnode[0].num_actions*5+2), dtype=np.float32) 
    cdef int i
    result[pnode[0].action] = 1. # action
    result[pnode[0].num_actions] = pnode[0].r # reward
    result[pnode[0].num_actions+1] = pnode[0].v # value
    for i in range(int(pnode[0].ppchildren[0].size())):
        child = pnode[0].ppchildren[0][i][0]
        result[pnode[0].num_actions+2+i] = child.logit # child_logits
        result[pnode[0].num_actions*2+2+i] = average(child.prollout_qs[0]) # child_rollout_qs_mean
        result[pnode[0].num_actions*3+2+i] = maximum(child.prollout_qs[0]) # child_rollout_qs_max
        result[pnode[0].num_actions*4+2+i] = child.rollout_n / <float>pnode[0].rec_t # child_rollout_ns_enc
    if detailed:
        pnode[0].max_q = (maximum(pnode[0].prollout_qs[0]) - pnode[0].r) / pnode[0].discounting
        result[pnode[0].num_actions*5+2] = pnode[0].trail_r / pnode[0].discounting
        result[pnode[0].num_actions*5+3] = pnode[0].rollout_q / pnode[0].discounting
        result[pnode[0].num_actions*5+4] = pnode[0].max_q
    return result

cdef node_del(Node* pnode, int except_idx):
    cdef int i
    del pnode[0].prollout_qs
    for i in range(int(pnode[0].ppchildren[0].size())):
        if i != except_idx:
            node_del(pnode[0].ppchildren[0][i], -1)
        else:
            pnode[0].ppchildren[0][i][0].pparent = NULL
    del pnode[0].ppchildren
    if pnode[0].encoded != NULL:
        Py_DECREF(<object>pnode[0].encoded)
    free(pnode)


cdef class cVecModelWrapper():
    """Wrap the gym environment with a model; output for each 
    step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out) that corresponds to underlying 
    environment frame and output from the model wrapper.
    """
    # setting
    cdef int rec_t
    cdef float discounting
    cdef float depth_discounting
    cdef bool perfect_model
    cdef bool tree_carry
    cdef int reward_type
    cdef bool reward_transform
    cdef bool actor_see_encode
    cdef int num_actions
    cdef int obs_n    
    cdef int env_n
    cdef bool time 

    # python object
    cdef object device
    cdef object env
    cdef object timings
    cdef readonly baseline_max_q
    cdef readonly baseline_mean_q    
    cdef readonly object model_out_shape
    cdef readonly object gym_env_out_shape

    # tree statistic
    cdef vector[Node*] cur_nodes
    cdef vector[Node*] root_nodes    
    cdef float[:] root_nodes_qmax
    cdef float[:] root_nodes_qmax_
    cdef int[:] rollout_depth
    cdef int[:] max_rollout_depth
    cdef int[:] cur_t

    # internal variables only used in step function
    cdef float[:] depth_delta
    cdef int[:] max_rollout_depth_
    cdef float[:] mean_q
    cdef float[:] max_q
    cdef int[:] status
    cdef vector[Node*] cur_nodes_
    cdef float[:] par_logits
    cdef float[:, :] full_reward
    cdef bool[:] full_done
    cdef bool[:] full_real_done

    def __init__(self, env, env_n, flags, device=None, time=False):
        assert flags.perfect_model, "imperfect model not yet supported"
        assert not flags.thres_carry, "thres_carry not yet supported"
        assert not flags.model_rnn, "model_rnn not yet supported"
        assert not flags.flex_t, "flexible time step not yet supported"

        self.device = torch.device("cpu") if device is None else device
        self.env = env     
        self.rec_t = flags.rec_t               
        self.discounting = flags.discounting
        self.depth_discounting = flags.depth_discounting
        self.tree_carry = flags.tree_carry
        self.num_actions = env.action_space[0].n
        self.reward_type = flags.reward_type
        self.reward_transform = flags.reward_transform
        self.actor_see_encode = flags.actor_see_encode
        self.env_n = env_n
        self.obs_n = 9 + self.num_actions * 10 + self.rec_t
        self.model_out_shape = (self.obs_n, 1, 1)
        self.gym_env_out_shape = env.observation_space.shape[1:]

        self.baseline_max_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)
        self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)        
        self.time = time
        self.timings = util.Timings()

        # internal variable init.
        self.depth_delta = np.zeros(self.env_n, dtype=np.float32)
        self.max_rollout_depth_ = np.zeros(self.env_n, dtype=np.intc)
        self.mean_q =  np.zeros(self.env_n, dtype=np.float32)
        self.max_q = np.zeros(self.env_n, dtype=np.float32)
        self.status = np.zeros(self.env_n, dtype=np.intc)
        self.par_logits = np.zeros(self.num_actions, dtype=np.float32)
        self.full_reward = np.zeros((self.env_n, 2 if self.reward_type == 1 else 1), dtype=np.float32)
        self.full_done = np.zeros(self.env_n, dtype=np.bool)
        self.full_real_done = np.zeros(self.env_n, dtype=np.bool)
        
    def reset(self, model_net):
        """reset the environment; should only be called in the initial"""
        cdef int i
        cdef Node* root_node
        cdef Node* cur_node
        cdef float[:,:] model_out        

        with torch.no_grad():
            # some init.
            self.root_nodes_qmax = np.zeros(self.env_n, dtype=np.float32)
            self.root_nodes_qmax_ = np.zeros(self.env_n, dtype=np.float32)
            self.rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.max_rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.cur_t = np.zeros(self.env_n, dtype=np.intc)

            # reset obs
            obs = self.env.reset()

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
            pass_action = torch.zeros(self.env_n, dtype=torch.long)
            _, vs, _, logits, model_encodes = model_net(obs_py, 
                                                pass_action.unsqueeze(0).to(self.device), 
                                                one_hot=False)  
            vs = vs.cpu()
            logits = logits.cpu()
            env_state = self.env.clone_state(inds=np.arange(self.env_n))

            # compute and update root node and current node
            for i in range(self.env_n):
                root_node = node_new(pparent=NULL, action=pass_action[i].item(), logit=0., num_actions=self.num_actions, 
                    discounting=self.discounting, rec_t=self.rec_t)                
                if not self.actor_see_encode:
                    encoded = {"env_state": env_state[i], "gym_env_out": obs_py[i]}
                else:
                    encoded = {"env_state": env_state[i], "gym_env_out": obs_py[i], "model_encodes": model_encodes[0,i]}
                node_expand(pnode=root_node, r=0., v=vs[-1, i].item(), done=False,
                    logits=logits[-1, i].numpy(), encoded=<PyObject*>encoded, override=False)
                node_visit(pnode=root_node)
                self.root_nodes.push_back(root_node)
                self.cur_nodes.push_back(root_node)
            
            # compute model_out
            model_out = self.compute_model_out(None, None)

            gym_env_out = []
            for i in range(self.env_n):
                encoded = <dict>self.cur_nodes[i][0].encoded
                gym_env_out.append(encoded["gym_env_out"].unsqueeze(0))
            gym_env_out = torch.concat(gym_env_out)

            if self.actor_see_encode:
                model_encodes = []
                for i in range(self.env_n):
                    encoded = <dict>self.cur_nodes[i][0].encoded
                    model_encodes.append(encoded["model_encodes"].unsqueeze(0))
                model_encodes = torch.concat(model_encodes)
            else:
                model_encodes = None

            # record initial root_nodes_qmax 
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q
            
            return torch.tensor(model_out, dtype=torch.float32, device=self.device), gym_env_out, model_encodes

    def step(self, action, model_net):  
        # action is tensor of shape (env_n, 3)
        # which corresponds to real_action, im_action, reset, term

        cdef int i, j, k
        cdef int[:] re_action
        cdef int[:] im_action
        cdef int[:] reset

        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_    
        cdef float[:,:] model_out        

        cdef vector[int] pass_inds_restore
        cdef vector[int] pass_inds_step
        cdef vector[int] pass_inds_reset
        cdef vector[int] pass_inds_reset_
        cdef vector[int] pass_action

        cdef float[:] vs
        cdef float[:,:] logits

        if self.time: self.timings.reset()
        action = action.cpu().int().numpy()
        re_action, im_action, reset = action[:, 0], action[:, 1], action[:, 2]

        pass_env_states = []

        for i in range(self.env_n):            
            # compute the mask of real / imagination step                             
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
            self.depth_delta[i] = self.depth_discounting ** self.rollout_depth[i]
            if self.cur_t[i] < self.rec_t - 1: # imagaination step
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(self.max_rollout_depth[i], self.rollout_depth[i])
                next_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if node_expanded(next_node):
                    self.status[i] = 2
                elif self.cur_nodes[i][0].done:
                    self.status[i] = 3
                else:
                    if self.status[i] != 0 or self.status[i] != 4: # no need restore if last step is real or just expanded
                        encoded = <dict> self.cur_nodes[i][0].encoded
                        pass_env_states.append(encoded["env_state"])
                        pass_inds_restore.push_back(i)
                        pass_action.push_back(im_action[i])
                        pass_inds_step.push_back(i)
                    self.status[i] = 4  
            else: # real step
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0          
                self.max_rollout_depth[i] = 0
                # record baseline before moving on
                self.baseline_mean_q[i] = average(self.root_nodes[i][0].prollout_qs[0]) / self.discounting
                self.baseline_max_q[i] = maximum(self.root_nodes[i][0].prollout_qs[0]) / self.discounting
                encoded = <dict> self.root_nodes[i][0].encoded
                pass_env_states.append(encoded["env_state"])
                pass_inds_restore.push_back(i)
                pass_action.push_back(re_action[i])
                pass_inds_step.push_back(i)
                self.status[i] = 1              
        if self.time: self.timings.time("misc_1")

        # restore env      
        if pass_inds_restore.size() > 0:
            self.env.restore_state(pass_env_states, inds=pass_inds_restore)
        # one step of env
        if pass_inds_step.size() > 0:
            obs, reward, done, info = self.env.step(pass_action, inds=pass_inds_step) 
            real_done = [m["real_done"] if "real_done" in m else done[n] for n, m in enumerate(info)]
        if self.time: self.timings.time("step_state")

        # reset needed?
        for i, j in enumerate(pass_inds_step):
            if self.status[j] == 1 and done[i]:
                pass_inds_reset.push_back(j)
                pass_inds_reset_.push_back(i) # index within pass_inds_step
        # reset
        if pass_inds_reset.size() > 0:
            obs_reset = self.env.reset(inds=pass_inds_reset) 
            for i, j in enumerate(pass_inds_reset_):
                obs[j] = obs_reset[i]
                pass_action[j] = 0            
        if self.time: self.timings.time("misc_2")

        # use model
        if pass_inds_step.size() > 0:
            with torch.no_grad():
                obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
                _, vs_, _, logits_, model_encodes = model_net(obs_py, 
                        torch.tensor(pass_action, dtype=long, device=self.device).unsqueeze(0), 
                        one_hot=False)  
            vs = vs_[-1].float().cpu().numpy()
            logits = logits_[-1].float().cpu().numpy()
            if self.time: self.timings.time("model")
            env_state = self.env.clone_state(inds=pass_inds_step)   
            if self.time: self.timings.time("clone_state")

        # compute the current and root nodes
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                # real transition
                new_root = (not self.tree_carry or 
                    not node_expanded(self.root_nodes[i][0].ppchildren[0][re_action[i]]) or done[j])
                if new_root:
                    root_node = node_new(pparent=NULL, action=pass_action[j], logit=0., num_actions=self.num_actions, 
                        discounting=self.discounting, rec_t=self.rec_t) 
                    if not self.actor_see_encode:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j]}
                    else:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j], "model_encodes": model_encodes[0,j]}
                    node_expand(pnode=root_node, r=0., v=vs[j], done=False,
                        logits=logits[j], encoded=<PyObject*>encoded, override=False)
                    node_del(self.root_nodes[i], except_idx=-1)
                    node_visit(root_node)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][re_action[i]]
                    if not self.actor_see_encode:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j]}
                    else:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j], "model_encodes": model_encodes[0,j]}
                    node_expand(pnode=root_node, r=0., v=vs[j], done=False,
                        logits=logits[j], encoded=<PyObject*>encoded, override=True)                        
                    node_del(self.root_nodes[i], except_idx=re_action[i])
                    node_visit(root_node)
                    
                j += 1
                root_nodes_.push_back(root_node)
                cur_nodes_.push_back(root_node)

            elif self.status[i] == 2:
                # expanded already
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)    

            elif self.status[i] == 3:
                # done already
                for k in range(self.num_actions):
                    self.par_logits[k] = self.cur_nodes[i].ppchildren[0][k][0].logit
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_expand(pnode=cur_node, r=0., v=0., done=True,
                        logits=self.par_logits, encoded=self.cur_nodes[i][0].encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)              
            
            elif self.status[i] == 4:
                # need expand
                if not self.actor_see_encode:
                    encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j]}
                else:
                    encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j], "model_encodes": model_encodes[0,j]}
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_expand(pnode=cur_node, r=reward[j], v=vs[j] if not done[j] else 0., done=done[j],
                        logits=logits[j], encoded=<PyObject*>encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)   
                j += 1                            

        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        if self.time: self.timings.time("compute_root_cur_nodes")

        # compute model_out        
        model_out = self.compute_model_out(action, self.status)

        gym_env_out = []
        for i in range(self.env_n):
            encoded = <dict>self.cur_nodes[i][0].encoded
            gym_env_out.append(encoded["gym_env_out"].unsqueeze(0))
        gym_env_out = torch.concat(gym_env_out)

        if self.actor_see_encode:
            model_encodes = []
            for i in range(self.env_n):
                encoded = <dict>self.cur_nodes[i][0].encoded
                model_encodes.append(encoded["model_encodes"].unsqueeze(0))
            model_encodes = torch.concat(model_encodes)
        else:
            model_encodes = None

        if self.time: self.timings.time("compute_model_out")

        # compute reward
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                self.full_reward[i][0] = reward[j]
            else:
                self.full_reward[i][0] = 0.
            if self.reward_type == 1:                        
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] != 1:                
                    self.full_reward[i][1] = (self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])*self.depth_delta[i]
                else:
                    self.full_reward[i][1] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1
        if self.time: self.timings.time("compute_reward")

        # compute done & full_real_done
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                self.full_done[i] = done[j]
                self.full_real_done[i] = real_done[j]
            else:
                self.full_done[i] = False
                self.full_real_done[i] = False
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1

        # compute reset
        for i in range(self.env_n):
            if reset[i]:
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5 # need to restore state on the next transition, so we need to alter the status from 4
        
        # some extra info
        info = {"cur_t": torch.tensor(self.cur_t, dtype=torch.long, device=self.device),
                "max_rollout_depth":  torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
                "real_done": torch.tensor(self.full_real_done, dtype=torch.bool, device=self.device)}
        if self.time: self.timings.time("end")

        return ((torch.tensor(model_out, dtype=torch.float32, device=self.device), gym_env_out, model_encodes), 
                torch.tensor(self.full_reward, dtype=torch.float32, device=self.device), 
                torch.tensor(self.full_done, dtype=torch.bool, device=self.device), 
                info)

    
    cdef float[:, :] compute_model_out(self, int[:, :]& action, int[:]& status):
        cdef int i
        cdef int idx1 = self.num_actions*5+5
        cdef int idx2 = self.num_actions*10+7

        result_np = np.zeros((self.env_n, self.obs_n), dtype=np.float32)
        cdef float[:, :] result = result_np        
        for i in range(self.env_n):
            result[i, :idx1] = node_stat(self.root_nodes[i], detailed=True)
            result[i, idx1:idx2] = node_stat(self.cur_nodes[i], detailed=False)    
            # reset
            if action is None or status[i] == 1:
                result[i, idx2] = 1.
            else:
                result[i, idx2] = action[i, 2]
            # time
            result[i, idx2+1+self.cur_t[i]] = 1.
            # deprec
            result[i, idx2+self.rec_t+1] = (self.discounting ** (self.rollout_depth[i]))           
        return result

    def close(self):
        cdef int i
        for i in range(self.env_n):
            node_del(self.root_nodes[i], except_idx=-1)
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