import time
import numpy as np
import argparse
import ray
import torch
from thinker.self_play import SelfPlayWorker
from thinker.learn import ActorLearner
from thinker.buffer import ActorBuffer, ParamBuffer
import thinker.util as util

if __name__ == "__main__":

    print("initializing")
    ray.init()
    st_time = time.time()
    flags = util.parse()
    
    actor_buffer = ActorBuffer.remote(batch_size=flags.batch_size)
    param_buffer = ParamBuffer.remote()

    self_play_workers = [SelfPlayWorker.remote(param_buffer, actor_buffer, n, flags) for n in range(flags.num_actors)]
    [x.gen_data.remote() for x in self_play_workers]    

    actor_learner = ActorLearner.remote(param_buffer, actor_buffer, 0, flags)
    ray.get(actor_learner.learn_data.remote())

    actor_buffer.set_finish.remote()
    print("time required: %fs" % (time.time()-st_time))

    