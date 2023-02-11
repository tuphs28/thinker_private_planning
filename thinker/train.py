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
    
    actor_buffer = ActorBuffer.remote(batch_size=flags.batch_size, num_p_actors=flags.num_p_actors)
    model_buffer = ModelBuffer.remote(flags) if flags.train_model else None
    param_buffer = GeneralBuffer.remote()        

    if flags.policy_type == PO_NET:
        policy_str = "actor network"
    elif flags.policy_type == PO_MODEL:
        policy_str = "base model network"
    else:
        raise Exception("policy not supported")

    num_gpus_available = torch.cuda.device_count()
    num_gpus_self_play = (num_gpus_available - flags.gpu_learn_actor * float(flags.train_actor) - 
        flags.gpu_learn_model * float(flags.train_model))
    num_self_play_gpu = num_gpus_self_play // flags.gpu_self_play
    logger.info("Number of self-play worker with GPU: %d/%d" % (num_self_play_gpu, flags.num_actors))
        
    logger.info("Starting %d actors with %s policy" % (flags.num_actors, policy_str))
    self_play_workers = [SelfPlayWorker.options(
        num_cpus=0, num_gpus=flags.gpu_self_play if n < num_self_play_gpu else 0).remote(
        param_buffer=param_buffer, 
        actor_buffer=actor_buffer, 
        model_buffer=model_buffer, 
        test_buffer=None,
        policy=flags.policy_type, 
        policy_params=None, 
        rank=n, 
        num_p_actors=flags.num_p_actors,
        flags=flags) for n in range(flags.num_actors)]
    r_worker = [x.gen_data.remote() for x in self_play_workers]    
    r_learner = []

    if flags.train_actor:
        actor_learner = ActorLearner.options(num_cpus=1, num_gpus=flags.gpu_learn_actor).remote(param_buffer, actor_buffer, 0, flags)
        r_learner.append(actor_learner.learn_data.remote())

    if flags.train_model:
        model_learner = ModelLearner.options(num_cpus=1, num_gpus=flags.gpu_learn_model).remote(param_buffer, model_buffer, 0, flags)
        r_learner.append(model_learner.learn_data.remote())

    if len(r_learner) >= 1: ray.get(r_learner)
    actor_buffer.set_finish.remote()    
    ray.get(r_worker)

    logger.info("time required: %fs" % (time.time()-st_time))

