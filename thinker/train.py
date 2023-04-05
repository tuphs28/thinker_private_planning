import time
import traceback
import sys
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
    if not hasattr(flags, 'cmd'):
        flags.cmd = ' '.join(sys.argv)    

    num_gpus_available = torch.cuda.device_count()
    num_cpus_available = ray.cluster_resources()['CPU']
    logger.info("Detected %d GPU %d CPU" % (num_gpus_available, num_cpus_available))


    if not flags.disable_auto_res:
        if flags.self_play_cpu:
            flags.gpu_num_actors = 0
            flags.gpu_num_p_actors = 0
            flags.gpu_self_play = 0          
            if num_gpus_available == 1:
                flags.gpu_learn_actor = 0.5
                flags.gpu_learn_model = 0.5
            else:
                flags.gpu_learn_actor = 1
                flags.gpu_learn_model = 1
        else: 

            flags.cpu_num_actors = 0 #int(num_cpus_available-8)
            flags.cpu_num_p_actors = 1

            if num_gpus_available == 1:
                flags.gpu_learn_actor = 0.5
                flags.gpu_learn_model = 0.25
                flags.gpu_self_play = 0.25
                flags.gpu_num_actors = 1
                flags.gpu_num_p_actors = 64

            elif num_gpus_available == 2:
                flags.gpu_learn_actor = 1
                flags.gpu_learn_model = 0.5
                flags.gpu_self_play = 0.25                   
                flags.gpu_num_actors = 2
                flags.gpu_num_p_actors = 32             

            elif num_gpus_available == 3:
                flags.gpu_learn_actor = 1
                flags.gpu_learn_model = 1
                flags.gpu_self_play = 0.5                               
                flags.gpu_num_actors = 2
                flags.gpu_num_p_actors = 32           
            elif num_gpus_available == 4:
                flags.gpu_learn_actor = 1
                flags.gpu_learn_model = 1
                flags.gpu_self_play = 1                              
                flags.gpu_num_actors = 2
                flags.gpu_num_p_actors = 32    
            if not flags.train_model:
                if num_gpus_available == 1:
                    flags.gpu_learn_model = 0
                    flags.gpu_num_actors = 2      
                if num_gpus_available == 2:  
                    flags.gpu_learn_model = 0
                    flags.gpu_self_play = 0.5
                if num_gpus_available == 3:  
                    flags.gpu_learn_model = 0
                    flags.gpu_self_play = 1
                if num_gpus_available == 4:
                    flags.gpu_num_actors = 3
                    flags.gpu_learn_model = 0            
                    flags.gpu_self_play = 1

    #flags.use_wandb = False
    #flags.model_warm_up_n = 6400
    #flags.model_buffer_n = 6400

    buffers = {
        "actor": ActorBuffer.options(num_cpus=1).remote(batch_size=flags.batch_size),
        "model": ModelBuffer.options(num_cpus=1).remote(flags) if flags.train_model else None,
        "actor_param": GeneralBuffer.options(num_cpus=1).remote(),
        "model_param": GeneralBuffer.options(num_cpus=1).remote(),
        "signal": GeneralBuffer.options(num_cpus=1).remote(),
        "test": None,
    }

    if flags.policy_type == PO_NET:
        policy_str = "actor network"
    elif flags.policy_type == PO_MODEL:
        policy_str = "base model network"
    else:
        raise Exception("policy not supported")

    num_gpus_available = torch.cuda.device_count()
    num_gpus_self_play = (num_gpus_available - flags.gpu_learn_actor * float(flags.train_actor) - 
        flags.gpu_learn_model * float(flags.train_model))
    
    if flags.gpu_self_play > 0:
        num_self_play_gpu = num_gpus_self_play // flags.gpu_self_play
        logger.info("Number of self-play worker with GPU: %d/%d" % (num_self_play_gpu, flags.gpu_num_actors))
    else:
        num_self_play_gpu = -1
        
    logger.info("Starting %d (gpu) self-play actors with %s policy" % (flags.gpu_num_actors, policy_str))
    logger.info("Starting %d (cpu) self-play actors with %s policy" % (flags.cpu_num_actors, policy_str))

    self_play_workers = []
    if flags.gpu_num_actors > 0:
        self_play_workers.extend([SelfPlayWorker.options(
            num_cpus=1, 
            num_gpus=flags.gpu_self_play).remote(
            buffers=buffers,    
            policy=flags.policy_type, 
            policy_params=None, 
            rank=n, 
            num_p_actors=flags.gpu_num_p_actors,
            flags=flags) for n in range(flags.gpu_num_actors)])
        
    if flags.cpu_num_actors > 0:
        self_play_workers.extend([SelfPlayWorker.options(
            num_cpus=1, 
            num_gpus=0).remote(
            buffers=buffers,    
            policy=flags.policy_type, 
            policy_params=None, 
            rank=n + flags.gpu_num_actors, 
            num_p_actors=flags.cpu_num_p_actors,
            flags=flags) for n in range(flags.cpu_num_actors)])
        
    r_worker = [x.gen_data.remote() for x in self_play_workers]    
    r_learner = []

    if flags.train_actor:
        actor_learner = ActorLearner.options(num_cpus=1, num_gpus=flags.gpu_learn_actor).remote(
            buffers, 0, flags)
        r_learner.append(actor_learner.learn_data.remote())

    if flags.train_model:
        model_learner = ModelLearner.options(num_cpus=1, num_gpus=flags.gpu_learn_model).remote(
            buffers, 0, flags)
        r_learner.append(model_learner.learn_data.remote())
      
    if len(r_learner) >= 1: ray.get(r_learner)
    buffers["actor"].set_finish.remote()    
    ray.get(r_worker)

    logger.info("time required: %fs" % (time.time()-st_time))

