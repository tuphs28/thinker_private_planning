import numpy as np
import ray

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
class ParamBuffer(object):
    def __init__(self):
        self.weights = {}

    def set_weights(self, name, weights):
        self.weights[name]  = weights
        return True

    def get_weights(self, name):        
        return self.weights[name] if name in self.weights else None