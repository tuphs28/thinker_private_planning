import numpy as np
import time
from operator import itemgetter 
import ray
import torch

AB_CAN_WRITE, AB_FULL, AB_FINISH = 0, 1, 2

@ray.remote
class ActorBuffer():
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.buffer = []
        self.finish = False
    
    def get_status(self):
        if self.finish: return AB_FINISH
        if len(self.buffer) >= 2 * self.batch_size: return AB_FULL
        return AB_CAN_WRITE

    def set_finish(self):
        self.finish = True

    def available_for_read(self):
        return len(self.buffer) >= self.batch_size
    
    def write(self, data):
        self.buffer.append(data[0])
    
    def read(self):
        if len(self.buffer) >= self.batch_size: 
            data = self.buffer[:self.batch_size]  
            del self.buffer[:self.batch_size]   
            return data 
            # remotely this becomes a ref of a list of ref; need double ray.get to get value
        else:
            return None

@ray.remote
class ModelBuffer():
    def __init__(self, flags):
        self.alpha = flags.priority_alpha
        
        self.t = flags.model_unroll_length   
        self.k = flags.model_k_step_return
        self.n = flags.actor_parallel_n             
        self.batch_mode = flags.model_batch_mode
        self.model_rnn = flags.model_rnn

        self.max_buffer_n = flags.model_buffer_n // (self.t * self.n) + 1 # maximum buffer length
        self.batch_size = flags.model_batch_size # batch size in returned sample
        self.wram_up_n = (flags.model_warm_up_n if not flags.load_checkpoint else
            flags.model_buffer_n)     # number of total transition before returning samples

        self.buffer = []
        self.state = []

        self.priorities = None
        self.next_inds = None
        self.cur_inds = np.full((flags.num_actors), fill_value=-1, dtype=np.int)

        self.base_ind = 0
        self.abs_tran_n = 0
        self.preload_n = 0
        self.clean_m = 0
    
    def write(self, data, state, rank):
        # data is a named tuple with elements of size (t+2*k-1, n, ...)        
        self.buffer.append(data)
        self.state.append(state)        

        p_shape = self.t*self.n if not self.batch_mode else self.n

        if self.priorities is None:
            self.priorities = np.ones((p_shape), dtype=float)
        else:
            max_priorities = np.full((p_shape), fill_value=self.priorities.max(), dtype=float)
            self.priorities = np.concatenate([self.priorities, max_priorities])

        # to record a table for chaining entry
        last_ind = int(self.cur_inds[rank] - self.base_ind / self.n)
        if last_ind >= 0:            
            self.next_inds[last_ind] = len(self.next_inds) + self.base_ind / self.n
        if self.next_inds is None:
            self.next_inds = np.full((1), fill_value=np.nan,)
        else:
            self.next_inds = np.concatenate([self.next_inds, np.full((1), fill_value=np.nan)])
        self.cur_inds[rank] = len(self.next_inds) + self.base_ind / self.n - 1
        

        self.abs_tran_n += self.t * self.n                        

        # clean periordically
        self.clean_m += 1
        if self.clean_m % 10 == 0:
            self.clean()
            self.clean_m = 0
    
    def read(self, beta):
        if self.priorities is None or self.abs_tran_n < self.wram_up_n: return None
        buffer_n = len(self.buffer)
        tran_n = len(self.priorities)
        probs = self.priorities ** self.alpha
        probs /= probs.sum()
        flat_inds = np.random.choice(tran_n, self.batch_size, p=probs, replace=False)

        if not self.batch_mode:
            inds = np.unravel_index(flat_inds, (buffer_n, self.t, self.n))
        else:
            inds = np.unravel_index(flat_inds, (buffer_n, self.n))

        weights = (tran_n * probs[flat_inds]) ** (-beta)
        weights /= weights.max()        

        data = []
        for d in range(len(self.buffer[0])):
            elems=[]
            for i in range(self.batch_size):
                if not self.batch_mode:
                    elems.append(self.buffer[inds[0][i]][d][inds[1][i]:inds[1][i]+2*self.k, inds[2][i]].unsqueeze(1))
                else:
                    elems.append(self.buffer[inds[0][i]][d][:-self.k, inds[1][i]].unsqueeze(1))
            data.append(torch.concat(elems, dim=1))
        data = type(self.buffer[0])(*data)        
        
        if self.model_rnn and self.batch_mode:
            data_state = []
            for d in range(len(self.state[0])):
                elems=[]
                for i in range(self.batch_size):                    
                    elems.append(self.state[inds[0][i]][d][inds[1][i]].unsqueeze(0))
                data_state.append(torch.concat(elems, dim=0))
            data_state = tuple(data_state)    
        else:
            data_state = None

        base_ind_pri = self.base_ind * self.t if not self.batch_mode else self.base_ind
        abs_flat_inds = flat_inds + base_ind_pri
        return data, data_state, weights, abs_flat_inds, self.abs_tran_n - self.preload_n

    def get_processed_n(self):
        return self.abs_tran_n - self.preload_n

    def update_priority(self, abs_flat_inds, priorities, state):
        """ Update priority and states in the buffer; both input 
        are np array of shape (update_size,)"""
        base_ind_pri = self.base_ind * self.t if not self.batch_mode else self.base_ind

        if not self.batch_mode:
            assert state is None, "can only update state in batch mode"
            # abs_flat_inds is an array of shape (model_batch_size,)
            # priorities is an array of shape (model_batch_size, k)
            priorities = priorities.transpose()

            flat_inds = abs_flat_inds - base_ind_pri # get the relative index
            mask = flat_inds >= 0 
            flat_inds = flat_inds[mask] 
            priorities = priorities[mask]

            flat_inds = flat_inds[:, np.newaxis] + np.arange(self.k) # flat_inds now stores uncarried indexes
            flat_inds_block = flat_inds // (self.t * self.n) # block index of flat_inds
            carry_mask = ~(flat_inds_block[:,[0]] == flat_inds_block).reshape(-1) 
            # if first index block is not the same as the later index block, we need to carry it

            flat_inds = flat_inds.reshape(-1)
            flat_inds_block = flat_inds_block.reshape(-1)
            carry_inds_block = self.next_inds[flat_inds_block[carry_mask]-1] - self.base_ind // self.n  # the correct index block

            flat_inds = flat_inds.astype(float)
            flat_inds[carry_mask] = flat_inds[carry_mask] + (-flat_inds_block[carry_mask] + carry_inds_block) * (self.t * self.n) 
            mask = ~np.isnan(flat_inds)
            flat_inds = flat_inds[mask].astype(int)

            priorities = priorities.reshape(-1)[mask]            
            self.priorities[flat_inds] = priorities      
        else:
            flat_inds = abs_flat_inds - base_ind_pri
            mask = (flat_inds > 0) & (flat_inds < len(self.priorities))
            self.priorities[flat_inds[mask]] = priorities[mask]        

            if state is not None:
                mask = (flat_inds >= 0)
                if len(mask) > 0:                
                    flat_inds_masked = flat_inds[mask]
                    state_masked = tuple(x[:, mask] for x in state) # assume batch size is in the second dim
                    inds = np.unravel_index(flat_inds_masked, (len(self.buffer), self.n))
                    for i in range(len(flat_inds_masked)):
                        next_abs_ind = self.next_inds[inds[0][i]]
                        if np.isnan(next_abs_ind): continue
                        next_ind = int(next_abs_ind - self.base_ind / self.n)
                        for d in range(len(self.state[0])):
                            self.state[next_ind][d][:, inds[1][i]] = state_masked[d][:, i]

    def clean(self):        
        buffer_n = len(self.buffer)
        if buffer_n > self.max_buffer_n:
            excess_n = buffer_n - self.max_buffer_n
            del self.buffer[:excess_n]
            del self.state[:excess_n]
            self.next_inds = self.next_inds[excess_n:]    
            if not self.batch_mode:
                self.priorities = self.priorities[excess_n * self.t * self.n:]                            
            else:
                self.priorities = self.priorities[excess_n * self.n:]    
            self.base_ind += excess_n * self.n                        

    def check_preload(self):
        return (len(self.buffer) >= self.max_buffer_n, len(self.buffer) * self.t * self.n)

    def set_preload(self):
        self.preload_n = self.abs_tran_n
        return True

@ray.remote
class GeneralBuffer(object):
    def __init__(self):
        self.data = {}
    
    def extend_data(self, name, x):
        if name in self.data:
            self.data[name].extend(x)
        else:
            self.data[name] = x
        return self.data[name]

    def update_dict_item(self, name, key, value):
        if not name in self.data: self.data[name] = {}
        self.data[name][key] = value
        return True        

    def set_data(self, name, x):
        self.data[name] = x
        return True

    def get_data(self, name):        
        return self.data[name] if name in self.data else None
