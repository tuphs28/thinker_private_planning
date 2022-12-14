import time
import numpy as np
import argparse
import ray
import torch
from thinker.self_play import SelfPlayWorker
from thinker.learn_actor import ActorLearner
from thinker.learn_model import ModelLearner
from thinker.buffer import ActorBuffer, ParamBuffer, ModelBuffer
import thinker.util as util

if __name__ == "__main__":

    print("Initializing...")
    ray.init()
    st_time = time.time()
    flags = util.parse()
    
    actor_buffer = ActorBuffer.remote(batch_size=flags.batch_size)
    model_buffer = ModelBuffer.remote(flags) if flags.train_model else None
    param_buffer = ParamBuffer.remote()        

    self_play_workers = [SelfPlayWorker.remote(param_buffer, actor_buffer, model_buffer, n, flags) for n in range(flags.num_actors)]
    [x.gen_data.remote() for x in self_play_workers]    

    actor_learner = ActorLearner.remote(param_buffer, actor_buffer, 0, flags)
    r_actor = actor_learner.learn_data.remote()

    if flags.train_model:
        model_learner = ModelLearner.remote(param_buffer, model_buffer, 0, flags)
        r_model = model_learner.learn_data.remote()
        ray.get([r_actor, r_model])
    else:
        ray.get(r_actor)
        
    actor_buffer.set_finish.remote()    
    print("time required: %fs" % (time.time()-st_time))

    