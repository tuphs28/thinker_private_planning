import time
import numpy as np
import argparse
import ray
import torch
from thinker.self_play import SelfPlayWorker, PO_NET, PO_MODEL
from thinker.learn_actor import ActorLearner
from thinker.learn_model import ModelLearner
from thinker.buffer import ActorBuffer, ModelBuffer, GeneralBuffer
import thinker.util as util

if __name__ == "__main__":

    print("Initializing...")
    ray.init()
    st_time = time.time()
    flags = util.parse()
    
    actor_buffer = ActorBuffer.remote(batch_size=flags.batch_size)
    model_buffer = ModelBuffer.remote(flags) if flags.train_model else None
    param_buffer = GeneralBuffer.remote()        

    if flags.policy_type == PO_NET:
        policy_str = "actor network"
    elif flags.policy_type == PO_MODEL:
        policy_str = "base model network"
    else:
        raise Exception("policy not supported")
        
    print("Starting %d actors with %s policy" % (flags.num_actors, policy_str))
    self_play_workers = [SelfPlayWorker.remote(param_buffer=param_buffer, 
      actor_buffer=actor_buffer, model_buffer=model_buffer, test_buffer=None,
      policy=flags.policy_type, rank=n, flags=flags) for n in range(flags.num_actors)]
    r_worker = [x.gen_data.remote() for x in self_play_workers]    

    r_learner = []
    if flags.train_actor:
        actor_learner = ActorLearner.remote(param_buffer, actor_buffer, 0, flags)
        r_learner.append(actor_learner.learn_data.remote())

    if flags.train_model:
        model_learner = ModelLearner.remote(param_buffer, model_buffer, 0, flags)
        r_learner.append(model_learner.learn_data.remote())

    if len(r_learner) >= 1: ray.get(r_learner)
    actor_buffer.set_finish.remote()    
    ray.get(r_worker)

    print("time required: %fs" % (time.time()-st_time))

    