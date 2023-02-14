
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

    logger = util.logger()

    logger.info("Initializing...")    

    ray.init()
    st_time = time.time()
    flags = util.parse()
    
    param_buffer = GeneralBuffer.remote()        
    test_buffer = GeneralBuffer.remote()      

    if flags.policy_type == PO_NET:
        policy_str = "actor network"
    elif flags.policy_type == PO_MODEL:
        policy_str = "base model network"
    else:
        raise Exception("policy not supported")

    flags.num_actors = 4
    flags.num_p_actors = 16
    flags.train_actor = False
    flags.train_model = False
    flags.preload_actor = flags.load_checkpoint+"/ckp_actor.tar"
    flags.preload_model = flags.load_checkpoint+"/ckp_model.tar"    
    flags.cwrapper = True

    test_eps_n = 200 // (flags.num_actors * flags.num_p_actors)
        
    logger.info("Starting %d actors with %s policy" % (flags.num_actors, policy_str))
    self_play_workers = [SelfPlayWorker.options(
        num_cpus=0, num_gpus=0.25).remote(
        param_buffer=param_buffer, 
        actor_buffer=None, 
        model_buffer=None, 
        test_buffer=test_buffer,
        policy=flags.policy_type, 
        policy_params=None, 
        rank=n, 
        num_p_actors=flags.num_p_actors,
        flags=flags) for n in range(flags.num_actors)]
    r_worker = [x.gen_data.remote(test_eps_n=test_eps_n) for x in self_play_workers]   
    ray.get(r_worker)

    logger.info("time required: %fs" % (time.time()-st_time))
    