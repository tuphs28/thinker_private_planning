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

        self.max_buffer_n = flags.model_buffer_n // (self.t * self.n) + 1 # maximum buffer length
        self.batch_size = flags.model_batch_size # batch size in returned sample
        self.wram_up_n = (flags.model_warm_up_n if not flags.load_checkpoint else
            flags.model_buffer_n)     # number of total transition before returning samples

        self.buffer = []
        self.priorities = None
        self.base_ind = 0
        self.abs_tran_n = 0
        self.preload_n = 0
        self.clean_m = 0
    
    def write(self, data):
        # data is a named tuple with elements of size (t+k, n, ...)        
        self.buffer.append(data)
        if self.priorities is None:
            self.priorities = np.ones((self.t * self.n), dtype=float)
        else:
            max_priorities = np.full((self.t * self.n), fill_value=self.priorities.max(), dtype=float)
            self.priorities = np.concatenate([self.priorities, max_priorities])
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
        inds = np.unravel_index(flat_inds, (buffer_n, self.t, self.n))

        weights = (tran_n * probs[flat_inds]) ** (-beta)
        weights /= weights.max()        

        data = []
        for d in range(len(self.buffer[0])):
            elems=[]
            for i in range(self.batch_size):
                elems.append(self.buffer[inds[0][i]][d][inds[1][i]:inds[1][i]+self.k+1, inds[2][i]].unsqueeze(1))
            data.append(torch.concat(elems, dim=1))
        data = type(self.buffer[0])(*data)

        abs_flat_inds = flat_inds + self.base_ind
        return data, weights, abs_flat_inds, self.abs_tran_n - self.preload_n

    def get_processed_n(self):
        return self.abs_tran_n - self.preload_n

    def update_priority(self, abs_flat_inds, priorities):
        """ Update priority in the buffer; both input 
        are np array of shape (update_size,)"""
        flat_inds = abs_flat_inds - self.base_ind
        mask = (flat_inds > 0)
        self.priorities[flat_inds[mask]] = priorities[mask]

    def clean(self):        
        buffer_n = len(self.buffer)
        if buffer_n > self.max_buffer_n:
            excess_n = buffer_n - self.max_buffer_n
            del self.buffer[:excess_n]
            self.priorities = self.priorities[excess_n * self.t * self.n:]
            self.base_ind += excess_n * self.t * self.n

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
