import collections
import time
import timeit
import argparse
import subprocess
import os 
import logging
import numpy as np
import torch

def parse(args=None, override=True):
    parser = argparse.ArgumentParser(description="Thinker v1")
    
    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")
    parser.add_argument("--savedir", default="~/RS/thinker/logs/thinker",
                        help="Root dir where experiment data will be saved.")

    # Environment settings
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use wandb logging")
    parser.add_argument("--env", type=str, default="cSokoban-v0",
                        help="Gym environment.")
    parser.add_argument("--cwrapper", action="store_true",
                        help="Whether to use C++ version of model wrapper")                                                
    parser.add_argument("--reward_clipping", default=-1, type=float, 
                       help="Reward clipping.")
    parser.add_argument("--reward_transform", action="store_true",
                        help="Whether to transform the reward as MuZero.")                       

    # Resources settings.
    parser.add_argument("--gpu_learn_actor", default=0.5, type=float,
                        help="Number of gpu per actor learning.") 
    parser.add_argument("--gpu_learn_model", default=0.5, type=float,
                        help="Number of gpu per model learning.") 
    parser.add_argument("--gpu_self_play", default=0.25, type=float,
                        help="Number of gpu per self-play worker.")     
    parser.add_argument("--float16",  action="store_true",
                        help="Whether to use float 16 precision in training.")                                                     
    parser.add_argument("--num_actors", default=48, type=int, 
                        help="Number of actors (default: 48).")
    parser.add_argument("--num_p_actors", default=1, type=int, 
                        help="Number of parallel env. per actor")                          

    # Preload settings.
    parser.add_argument("--load_checkpoint", default="",
                        help="Load checkpoint directory.")    
    parser.add_argument("--preload_actor", default="",
                        help="File location of the preload actor network.")                        
    parser.add_argument("--preload_model", default="",
                        help="File location of the preload model network.")                                     

    # Actor Training settings.            
    parser.add_argument("--policy_type", default=0, type=int, 
                        help="Policy used for self-play worker; 0 for actor net, 1 for model policy, 2 for 1-step greedy") 
    parser.add_argument("--disable_train_actor", action="store_false", dest="train_actor",
                        help="Disable training of actor.")   
    parser.add_argument("--total_steps", default=50000000, type=int, 
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=32, type=int, 
                        help="Actor learner batch size.")
    parser.add_argument("--unroll_length", default=200, type=int, 
                        help="The unroll length (time dimension).")
    parser.add_argument("--actor_warm_up_n", default=0, type=int, 
                        help="Number of transition accumulated before actor start learning.")                                                                        
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")

    # Model Training settings. 
    parser.add_argument("--disable_train_model", action="store_false", dest="train_model",
                        help="Disable training of model.")
    parser.add_argument("--model_batch_size", default=128, type=int, 
                        help="Model learner batch size.")   
    parser.add_argument("--model_batch_mode", action="store_true",
                        help="Whether to use the full rollout from model buffer in training.")                                            
    parser.add_argument("--model_unroll_length", default=200, type=int, 
                        help="Number of transition per unroll in model buffer.")
    parser.add_argument("--model_k_step_return", default=5, type=int, 
                        help="Number of recurrent step when training the model.")    
    parser.add_argument("--priority_alpha", default=0.6, type=float,
                        help="Alpha used to compute the priority from model buffer; 0. for no priority replay.")
    parser.add_argument("--priority_beta", default=0.4, type=float,
                        help="Initial beta used to compute the priority from model buffer.")
    parser.add_argument("--priority_type", default=2, type=int,
                        help="Type 0: update priority for all time step; Type 1: update priority for the first time step (mean of all error); Type 1: update priority for the first time step (first-step error)")                        
    parser.add_argument("--model_buffer_n", default=200000, type=int, 
                        help="Maximum number of transition in model buffer.") 
    parser.add_argument("--model_warm_up_n", default=400000, type=int, 
                        help="Number of transition accumulated before model start learning.")     
    parser.add_argument("--test_policy_type", default=1, type=int, 
                        help="Policy used for testing model; 0 for actor net, 1 for model policy, 2 for 1-step greedy")                         
    parser.add_argument("--model_min_step_per_transition", default=14, type=int, 
                        help="Minimum number of model learning step on one transition")                         
    parser.add_argument("--model_max_step_per_transition", default=15, type=int, 
                        help="Maximum number of model learning step on one transition")                                                 
                            
  
    # Actor architecture settings
    parser.add_argument("--tran_dim", default=96, type=int, 
                        help="Size of transformer hidden dim.")
    parser.add_argument("--tran_mem_n", default=40, type=int, 
                        help="Size of transformer memory.")
    parser.add_argument("--tran_layer_n", default=3, type=int,
                        help="Number of transformer layer.")
    parser.add_argument("--tran_t", default=1, type=int, 
                        help="Number of recurrent step for transformer.")   
    parser.add_argument("--tran_lstm_no_attn", action="store_true",
                        help="Whether to disable attention in LSTM-transformer.")
    parser.add_argument("--tran_attn_b", default=5,
                        type=float, help="Bias attention for current position.")        
    parser.add_argument("--actor_see_p", default=0,
                        type=float, help="Probability of allowing actor to see state.")                                
    parser.add_argument("--actor_see_encode",  action="store_true",
                        help="Whether to see the encoded state")        
    parser.add_argument("--actor_see_double_encode",  action="store_true",
                        help="Whether to see double encoded state (need to be eanabled with actor_see_encode)")                                          
    parser.add_argument("--actor_drc", action="store_true",
                        help="Whether to use drc in encoding state")    

    # Model architecure settings
    parser.add_argument("--model_type_nn", default=0,
                        type=float, help="Model type.")          
    parser.add_argument("--model_size_nn", default=1,
                        type=int, help="Model size multipler.")       
    parser.add_argument("--model_zero_init", action="store_true",
                        help="Zero initialisation for the model output")            
                                                           
    
    # Actor loss settings
    parser.add_argument("--entropy_cost", default=0.00001,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--im_entropy_cost", default=0.000005,
                        type=float, help="Imagainary Entropy cost/multiplier.")         
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--reg_cost", default=0.01,
                        type=float, help="Reg cost/multiplier.")
    parser.add_argument("--real_cost", default=1,
                        type=float, help="Real reward - real action cost/multiplier.")      
    parser.add_argument("--real_im_cost", default=1,
                        type=float, help="Real reward - imagainary action cost/multiplier.")          
    parser.add_argument("--im_cost", default=1,
                        type=float, help="Imaginary reward cost/multiplier.")   
    parser.add_argument("--im_cost_anneal", action="store_true", 
                        help="Whether to anneal im_cost to zero.")                        
    parser.add_argument("--reset_no_im_cost", action="store_true",
                        help="Whether to disable training reset action by planning reward")                            
    parser.add_argument("--discounting", default=0.97,
                        type=float, help="Discounting factor.")
    parser.add_argument("--lamb", default=1,
                        type=float, help="Lambda when computing trace.")
    parser.add_argument("--rnn_grad_scale", default=1.,
                        type=float, help="Gradient scale for RNN in actor network.")
                        
    
    # Model loss settings
    parser.add_argument("--model_logits_loss_cost", default=0.5, type=float,
                       help="Multipler to policy logit loss when training the model.")                            
    parser.add_argument("--model_vs_loss_cost", default=1, type=float,
                       help="Multipler to vs loss when training the model.")                           
    parser.add_argument("--model_rs_loss_cost", default=1, type=float,
                       help="Multipler to rs loss when training the model.")                                                  
    parser.add_argument("--model_sup_loss_cost", default=1, type=float,
                       help="Multipler to self-supervise loss when training the model.")                                                                         
    parser.add_argument("--model_bootstrap_type", default=0, type=int,
                       help="0 for mean root value, 1 for max root value, 2 for actor'svalue.")     
    parser.add_argument("--model_supervise", action="store_true",
                        help="Whether to add self-supervised loss in model training")    
    parser.add_argument("--model_supervise_type", default=0, type=int,
                       help="0 for efficientZero, 1 for direct cosine similarity.")                            


    # Model wrapper settings
    parser.add_argument("--reward_type", default=1, type=int, 
                        help="Reward type")     
    parser.add_argument("--im_gate", action="store_true", 
                        help="Whether to gate im reward by real advantage.")            
    parser.add_argument("--disable_perfect_model", action="store_false", dest="perfect_model",
                        help="Whether to use perfect model.")          
    parser.add_argument("--rec_t", default=20, type=int, 
                        help="Number of planning steps.")
    parser.add_argument("--disable_tree_carry", action="store_false", dest="tree_carry",
                        help="Whether to carry over the tree.")   
    parser.add_argument("--depth_discounting", default=1.,
                        type=float, help="Discounting factor for planning reward based on search depth.")                           
    parser.add_argument("--max_depth", default=-1,
                        type=int, help="Maximal search death.")                                                   

    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.0002,
                        type=float, help="Learning rate for actor learne.")
    parser.add_argument("--model_learning_rate", default=0.00002,
                        type=float, help="Learning rate for model learner.")                        
    parser.add_argument("--grad_norm_clipping", default=600, type=float,
                        help="Global gradient norm clip for actor learner.")
    parser.add_argument("--model_grad_norm_clipping", default=0, type=float,
                        help="Global gradient norm clip for model learner.")            
    
    if args is None:
        flags = parser.parse_args()  
    else:
        flags = parser.parse_args(args)  

    assert not (not flags.perfect_model and flags.actor_see_p > 0 and not flags.actor_see_encode), "learned model cannot see gym_env_out directly"
    
    fs = ["load_checkpoint", "savedir", "preload_model", "preload_actor"]    
    for f in fs:
        path = getattr(flags, f)
        if path: setattr(flags, f, os.path.expanduser(path))
            
    if flags.load_checkpoint and override:
        check_point_path = os.path.join(flags.load_checkpoint, "ckp_actor.tar")
        train_checkpoint = torch.load(check_point_path, torch.device("cpu"))
        flags_ = train_checkpoint["flags"]
        for k, v in flags_.items(): 
            if k not in ["load_checkpoint", "policy_type"]: setattr(flags, k, v)

    if flags.xpid is None:
        flags.xpid = "thinker-%s" % time.strftime("%Y%m%d-%H%M%S")

    return flags

def tuple_map(x, f):
    if type(x) == tuple:
        return tuple(f(y) if y is not None else None for y in x)
    else:
        return type(x)(*(f(y) if y is not None else None for y in x))

def construct_tuple(x, **kwargs):
    return x(**{k: kwargs[k] if k in kwargs else None for k in x._fields})

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def decode_model_out(model_out, num_actions, reward_tran):
    idx1 = num_actions*5+5
    idx2 = num_actions*10+7
    d = dec if reward_tran else lambda x: x
    return {
        "root_action": model_out[0,:,:num_actions],
        "root_r": d(model_out[0,:,[num_actions]]),
        "root_v": d(model_out[0,:,[num_actions+1]]),
        "root_logits": model_out[0,:,num_actions+2:2*num_actions+2],
        "root_qs_mean": d(model_out[0,:,2*num_actions+2:3*num_actions+2]),
        "root_qs_max": d(model_out[0,:,3*num_actions+2:4*num_actions+2]),
        "root_ns": model_out[0,:,4*num_actions+2:5*num_actions+2],
        "root_trail_r": d(model_out[0,:,[5*num_actions+2]]),
        "root_trail_q": d(model_out[0,:,[5*num_actions+3]]),
        "root_max_v": d(model_out[0,:,[5*num_actions+4]]),
        "cur_action": model_out[0,:,idx1:idx1+num_actions],
        "cur_r": d(model_out[0,:,[idx1+num_actions]]),
        "cur_v": d(model_out[0,:,[idx1+num_actions+1]]),
        "cur_logits": model_out[0,:,idx1+num_actions+2:idx1+2*num_actions+2],
        "cur_qs_mean": d(model_out[0,:,idx1+2*num_actions+2:idx1+3*num_actions+2]),
        "cur_qs_max": d(model_out[0,:,idx1+3*num_actions+2:idx1+4*num_actions+2]),
        "cur_ns": model_out[0,:,idx1+4*num_actions+2:idx1+5*num_actions+2],
        "reset": model_out[0,:,idx2],
        "time": model_out[0,:,idx2+1:-1],
        "derec": model_out[0,:,[-1]],
    }

def enc(x):
    return np.sign(x)*(np.sqrt(np.abs(x)+1)-1)+(0.001)*x    

def dec(x):
    return np.sign(x)*(np.square((np.sqrt(1+4*0.001*(torch.abs(x)+1+0.001))-1)/(2*0.001)) - 1) 

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def logger():
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("logs/out")
    if not logger.hasHandlers():
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        logger.addHandler(shandle)    
    logger.setLevel(logging.INFO)  
    return logger

class Timings:

    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result

class Wandb():
    def __init__(self, flags, subname=''):
        import wandb
        self.wandb = wandb
        exp_name = flags.xpid + subname
        self.wandb.init(
            project='thinker',
            config=flags,
            entity=os.getenv('WANDB_USER', 'stephen-chung'),
            reinit=True,
            # Restore parameters
            resume="allow",
            id=exp_name,
            name=exp_name,
        )
        self.wandb.config.update(flags, allow_val_change=True)

def compute_grad_norm(parameters, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm