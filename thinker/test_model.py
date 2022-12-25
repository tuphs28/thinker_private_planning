from collections import deque
import time
import timeit
import numpy as np
import argparse
import ray
import torch
from thinker.self_play import SelfPlayWorker, PO_NET, PO_MODEL, PO_NSTEP
from thinker.self_play import TrainActorOut
from thinker.buffer import ActorBuffer, ModelBuffer, GeneralBuffer
import thinker.util as util

if __name__ == "__main__":

    print("Initializing...")
    ray.init()
    st_time = time.time()
    flags = util.parse() 
    flags.train_actor = False
    flags.train_model = False    
    
    param_buffer = GeneralBuffer.remote()        
    test_buffer = GeneralBuffer.remote()        

    self_play_workers = [SelfPlayWorker.remote(
      param_buffer=param_buffer, 
      actor_buffer=None, 
      model_buffer=None, 
      test_buffer=test_buffer, 
      policy=PO_NSTEP, 
      policy_params={"n":2, "temp": 1/5},
      rank=n, 
      flags=flags) for n in range(flags.num_actors)]
    r_worker = [x.gen_data.remote(test_eps_n=1000) for x in self_play_workers]    
    ray.get(r_worker)
        
    print("time required: %fs" % (time.time()-st_time))



    