__version__ = '0.62'

import collections
import time
import timeit
import argparse
import subprocess
import os 
import logging
from matplotlib import pyplot as plt
import numpy as np
import torch
import re

def get_parser():
    parser = argparse.ArgumentParser(description=f"Thinker v{__version__}")
    
    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")
    parser.add_argument("--savedir", default="~/RS/thinker/logs/thinker",
                        help="Root dir where experiment data will be saved.")

    # Logging settings
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use wandb logging")
    parser.add_argument("--wandb_ckp_freq", type=int, default=500000,
                        help="Checkpoint frequency of wandb (in real steps) (-1 for not logging).")  
    parser.add_argument("--policy_vis_freq", type=int, default=500000,
                        help="Visualization frequency of wandb (in real steps) (-1 for not logging).")    
    parser.add_argument("--policy_vis_length", type=int, default=20,
                        help="Length of visualization (in real steps).")  

    # Environment settings
    parser.add_argument("--env", type=str, default="cSokoban-v0",
                        help="Gym environment.")
    parser.add_argument("--cwrapper", action="store_true",
                        help="Whether to use C++ version of model wrapper")                                                
    parser.add_argument("--reward_clipping", default=-1, type=float, 
                       help="Reward clipping.")                      

    # Resources settings.
    parser.add_argument("--disable_auto_res", action="store_true",
                        help="Whether to allocate resources automatically")    
    parser.add_argument("--self_play_cpu", action="store_true",
                        help="Whether to use cpu for self-play actors.")   
    parser.add_argument("--gpu_learn_actor", default=0.5, type=float,
                        help="Number of gpu per actor learning.") 
    parser.add_argument("--gpu_learn_model", default=0.5, type=float,
                        help="Number of gpu per model learning.") 
    parser.add_argument("--gpu_self_play", default=0.25, type=float,
                        help="Number of gpu per self-play worker.")     
    parser.add_argument("--float16",  action="store_true",
                        help="Whether to use float 16 precision in training.")                                                     
    parser.add_argument("--gpu_num_actors", default=1, type=int, 
                        help="Number of self-play actor (gpu)")
    parser.add_argument("--gpu_num_p_actors", default=32, type=int, 
                        help="Number of env per self-play actor (gpu)")                          
    parser.add_argument("--cpu_num_actors", default=32, type=int, 
                        help="Number of self-play actor (cpu)")
    parser.add_argument("--cpu_num_p_actors", default=2, type=int, 
                        help="Number of env per self-play actor (cpu)")                          
    parser.add_argument("--self_play_merge", action="store_true",
                        help="Whether to merge self-play and learn_model processes")      
    parser.add_argument("--profile", action="store_true",
                        help="Whether to do profiling")   
    
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
    parser.add_argument("--unroll_length", default=99, type=int, 
                        help="The unroll length (time dimension).")
    parser.add_argument("--actor_warm_up_n", default=0, type=int, 
                        help="Number of transition accumulated before actor start learning.")      
    parser.add_argument("--return_norm_type", default=1, type=int, 
                        help="Return norm type; -1 for no normalization, 0 for value normalization, 1 for advantage normalization")           
    parser.add_argument("--return_norm_b", default=1, type=float, 
                        help="Minimum normalization coef")  
    parser.add_argument("--return_norm_decay", default=0.99, type=float, 
                        help="Return normalization decay rate")     
    parser.add_argument("--return_norm_per", default=0.95, type=float, 
                        help="Percentile used in return normalization")                       
    parser.add_argument("--return_norm_buffer_n", default=100000, type=int, 
                        help="Normalization buffer; only used when it is larger than 0.") 
    parser.add_argument("--return_norm_use_std", action="store_true",
                        help="Use std instead of percentile in return norm.")                       
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")

    # Model Training settings. 
    parser.add_argument("--disable_train_model", action="store_false", dest="train_model",
                        help="Disable training of model.")
    parser.add_argument("--model_batch_size", default=128, type=int, 
                        help="Model learner batch size.")                                             
    parser.add_argument("--model_unroll_length", default=50, type=int, 
                        help="Number of transition per unroll in model buffer.")
    parser.add_argument("--model_k_step_return", default=5, type=int, 
                        help="Number of recurrent step when training the model.")    
    parser.add_argument("--priority_alpha", default=0.6, type=float,
                        help="Alpha used to compute the priority from model buffer; 0. for no priority replay.")
    parser.add_argument("--priority_beta", default=0.4, type=float,
                        help="Initial beta used to compute the priority from model buffer.")
    parser.add_argument("--priority_type", default=2, type=int,
                        help="Type 0: update priority for all time step; Type 1: update priority for the first time step (mean of all error); Type 2: update priority for the first time step (first-step error)")                        
    parser.add_argument("--model_buffer_n", default=200000, type=int, 
                        help="Maximum number of transition in model buffer.") 
    parser.add_argument("--model_warm_up_n", default=200000, type=int, 
                        help="Number of transition accumulated before model start learning.")     
    parser.add_argument("--test_policy_type", default=-1, type=int, 
                        help="Policy used for testing model; -1 for no testing, 0 for actor net, 1 for model policy, 2 for 1-step greedy")                         
    parser.add_argument("--model_min_step_per_transition", default=5, type=float, 
                        help="Minimum number of model learning step on one transition")                         
    parser.add_argument("--model_max_step_per_transition", default=6, type=float, 
                        help="Maximum number of model learning step on one transition")                                                 
                            
  
    # Actor architecture settings
    parser.add_argument("--actor_net_ver", default=1, type=int, 
                        help="Version of actor net.")    
    parser.add_argument("--tran_dim", default=128, type=int, 
                        help="Size of transformer hidden dim.")
    parser.add_argument("--tran_mem_n", default=40, type=int, 
                        help="Size of transformer memory.")
    parser.add_argument("--tran_layer_n", default=4, type=int,
                        help="Number of transformer layer.")
    parser.add_argument("--tran_t", default=1, type=int, 
                        help="Number of recurrent step for transformer.")   
    parser.add_argument("--tran_lstm_no_attn", action="store_true",
                        help="Whether to disable attention in LSTM-transformer.")
    parser.add_argument("--tran_attn_b", default=5,
                        type=float, help="Bias attention for current position.")        
    parser.add_argument("--actor_see_type", default=2, type=int,
                        help="What actor sees from model: -1 for nothing, 0. for predicted / true frame, 1. for z, 2. for h.")      
    parser.add_argument("--actor_see_double_encode",  action="store_true",
                        help="Whether to see double encoded state (need to be eanabled with actor_see_encode)")        
    parser.add_argument("--actor_drc", action="store_true",
                        help="Whether to use drc in encoding state")    
    parser.add_argument("--actor_encode_concat_type",  default=0,
                        type=int, help="Type of concating the encoding to model's output.")          
    
    # Critic architecture settings
    parser.add_argument("--critic_zero_init", action="store_true", dest="critic_zero_init",
                        help="Zero initialisation for the critic's output")      
    parser.add_argument("--critic_enc_type",  default=4,
                        type=int, help="Reward / value encoding type for the critic; \
                            0 for no encoding, \
                            1 for scalar encoding, \
                            2 for unbiased vector encoding, \
                            3 for biased vector encoding \
                            4 for moving scalar multiplication")      
    parser.add_argument("--critic_norm_b", default=1, type=float, 
                        help="Minimum normalization coef for critic; only used when critic_enc_type in [4, 5]") 
    

    # Model architecure settings
    parser.add_argument("--model_type_nn", default=0,
                        type=int, help="Model type.")          
    parser.add_argument("--model_size_nn", default=1,
                        type=int, help="Model size multipler.")      
    parser.add_argument("--model_downscale_c", default=2.,
                        type=float, help="Coefficient for downscaling number of channels")      
    parser.add_argument("--disable_model_zero_init", action="store_false", dest="model_zero_init",
                        help="Zero initialisation for the model output")  
    parser.add_argument("--model_enable_bn", action="store_false", dest="model_disable_bn",
                        help="Whether to disable batch norm in dynamic and output network")     
    parser.add_argument("--value_prefix", action="store_true",
                        help="Whether to use value prefix in model")         
    parser.add_argument("--disable_duel_net", action="store_false", dest="duel_net",
                        help="Whether to use duel net as model")            
    parser.add_argument("--frame_copy", action="store_true",
                        help="Whether to copy the last three frames in frame prediction") 
    parser.add_argument("--model_enc_type",  default=2, type=int, 
                        type=int, help="Reward / value encoding type for the model; \
                            0 for no encoding, \
                            2 for unbiased vector encoding, \
                            3 for biased vector encoding")       
                                                          
    
    # Actor loss settings
    # Real reward setting
    parser.add_argument("--entropy_cost", default=0.0003,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--entropy_type", default=1,
                        type=int, help="Entropy mask type; 0 for no masking; 1 for masking.")      
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--reg_cost", default=0.001,
                        type=float, help="Reg cost/multiplier.")
    parser.add_argument("--real_cost", default=1,
                        type=float, help="Real reward - real action cost/multiplier.")          
    parser.add_argument("--real_im_cost", default=1,
                        type=float, help="Real reward - imagainary action cost/multiplier.")          
    # Planning reward setting
    parser.add_argument("--im_entropy_cost", default=0.0003,
                        type=float, help="Imagainary Entropy cost/multiplier.")         
    parser.add_argument("--im_cost", default=1,
                        type=float, help="Imaginary reward cost/multiplier. 0 for no imaginary reward.")   
    parser.add_argument("--disable_im_anneal", action="store_false", dest="im_cost_anneal",
                        help="Whether to anneal im_cost to zero.")       
    parser.add_argument("--reset_no_im_cost", action="store_true",
                        help="Whether to disable training reset action by planning reward")     
    # Curiosity reward settings             
    parser.add_argument("--cur_cost", default=0,
                        type=float, help="Curiosity reward cost/multiplier. 0 for no curiosity.")   
    parser.add_argument("--cur_im_cost", default=0,
                        type=float, help="Curiosity reward - imagainary action cost/multiplier.")      
    parser.add_argument("--cur_reward_cost", default=0,
                        type=float, help="Coefficient of reward loss for computing curiosity reward.")   
    parser.add_argument("--cur_v_cost", default=0,
                        type=float, help="Coefficient of value loss for computing curiosity reward.")   
    parser.add_argument("--cur_enc_cost", default=0,
                        type=float, help="Coefficient of encoding loss for computing curiosity reward.")       
    parser.add_argument("--disable_cur_anneal", action="store_false", dest="cur_cost_anneal",
                        help="Whether to anneal cur_cost to zero.")
    parser.add_argument("--cur_done_gate", action="store_true", 
                        help="Whether to give no curisotiy reward when done.")
    parser.add_argument("--only_cur", action="store_true", 
                        help="Let's replace agents' real reward with curiosity reward.")
    parser.add_argument("--cur_warm_up_n",  default=1000000,
                        type=int, help="The real step that begins giving curiosity reward.")
         
    # Other reward-related settings        
    parser.add_argument("--discounting", default=0.97,
                        type=float, help="Discounting factor.")
    parser.add_argument("--lamb", default=1,
                        type=float, help="Lambda when computing trace.")
    parser.add_argument("--rnn_grad_scale", default=1.,
                        type=float, help="Gradient scale for RNN in actor network.")
                        
    
    # Model loss settings
    parser.add_argument("--model_logits_loss_cost", default=0.5, type=float,
                       help="Multipler to policy logit loss when training the model.")                            
    parser.add_argument("--model_vs_loss_cost", default=0.25, type=float,
                       help="Multipler to vs loss when training the model.")                           
    parser.add_argument("--model_rs_loss_cost", default=1., type=float,
                       help="Multipler to rs loss when training the model.")                                                  
    parser.add_argument("--model_done_loss_cost", default=1., type=float,
                       help="Multipler to done loss when training the model.")                                                      
    parser.add_argument("--model_sup_loss_cost", default=0, type=float,
                       help="Multipler to self-supervise loss when training the model.")                                                                         
    parser.add_argument("--model_img_loss_cost", default=10.,  type=float,
                       help="Multipler to image reconstruction loss when training the model.")     
    parser.add_argument("--model_reg_loss_cost", default=0, type=float,
                       help="Multipler to L2 reg loss for predictor encoding when training the model.")                                                                             
    parser.add_argument("--model_bootstrap_type", default=0, type=int,
                       help="0 for mean root value, 1 for max root value, 2 for actor's value.")    
    parser.add_argument("--model_supervise_type", default=1, type=int,
                       help="0 for efficientZero, 1 for direct cosine similarity, 2 for direct L2 loss, 3 for KL div.")                                                          
    parser.add_argument("--model_img_type", default=1, type=int,
                       help="0 for L2 loss on frame; 1 for mapped L2 loss on frame.")                                                          
    

    # Model wrapper settings            
    parser.add_argument("--perfect_model", action="store_true",
                        help="Whether to use perfect model.")          
    parser.add_argument("--rec_t", default=20, type=int, 
                        help="Number of planning steps.")
    parser.add_argument("--disable_tree_carry", action="store_false", dest="tree_carry",
                        help="Whether to carry over the tree.")   
    parser.add_argument("--depth_discounting", default=1.,
                        type=float, help="Discounting factor for planning reward based on search depth.")                           
    parser.add_argument("--max_depth", default=5,
                        type=int, help="Maximal search death.")                                                   

    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.0002,
                        type=float, help="Learning rate for actor learne.")
    parser.add_argument("--model_learning_rate", default=0.0001,
                        type=float, help="Learning rate for model learner.")                        
    parser.add_argument("--grad_norm_clipping", default=600, type=float,
                        help="Global gradient norm clip for actor learner.")
    parser.add_argument("--model_grad_norm_clipping", default=0, type=float,
                        help="Global gradient norm clip for model learner.") 

    # Misc
    parser.add_argument("--splay_model_update_freq", default=1, type=int,
                        help="Model update frequency for self-play process.")         
    parser.add_argument("--splay_actor_update_freq", default=1, type=int,
                        help="Actor update frequency for self-play process.")             
    parser.add_argument("--lmodel_model_update_freq", default=1, type=int,
                        help="Model update frequency for model learner.")   
    return parser      

def parse(args=None, override=True):
    parser = get_parser()    
    if args is None:
        flags = parser.parse_args()  
    else:
        flags = parser.parse_args(args)  
    return process_flags(flags, override)

def process_flags(flags, override=True):
    if flags.perfect_model:
        if flags.duel_net:
            print("Automatically disable duel net (perfect model).")
            flags.duel_net = False
        if flags.cur_cost > 0.:
            print("Automatically disable curiosity (perfect model).")
            flags.cur_cost = 0
        if flags.model_img_loss_cost > 0:
            print("Automatically setting  model_img_loss_cost = 0 (perfect model).")
            flags.model_img_loss_cost = 0
        if flags.model_sup_loss_cost > 0:
            print("Automatically setting  model_sup_loss_cost = 0 (perfect model).")
            flags.model_sup_loss_cost = 0
        if flags.model_done_loss_cost > 0:
            print("Automatically setting  model_done_loss_cost = 0 (perfect model).")
            flags.model_done_loss_cost = 0
    
    assert not (not flags.perfect_model and not flags.duel_net and flags.actor_see_type == 0), (
        "to see the frame, either use perfect model or duel net")
    assert flags.actor_net_ver == 1, "only actor_net_ver 1 is supported"

    fs = ["load_checkpoint", "savedir", "preload_model", "preload_actor"]    
    for f in fs:
        path = getattr(flags, f)
        if path: 
            path = os.path.abspath(os.path.expanduser(path))
            if os.path.islink(path):
                path = os.readlink(path)
            setattr(flags, f, path)
            
    if flags.load_checkpoint and override:
        check_point_path = os.path.join(flags.load_checkpoint, "ckp_actor.tar")
        train_checkpoint = torch.load(check_point_path, torch.device("cpu"))
        flags_ = train_checkpoint["flags"]
        for k, v in flags_.items(): 
            if k not in ["load_checkpoint", "policy_type"]: setattr(flags, k, v)

    if flags.xpid is None:
        flags.xpid = "thinker-%s" % time.strftime("%Y%m%d-%H%M%S")

    flags.__version__ = __version__ 
    return flags

def tuple_map(x, f):
    if type(x) == tuple:
        return tuple(f(y) if y is not None else None for y in x)
    else:
        return type(x)(*(f(y) if y is not None else None for y in x))

def safe_view(x, dims):
    if x is None:
        return None
    else:
        return x.view(*dims)

def safe_squeeze(x, dim=0):
    if x is None: 
        return None
    else:
        return x.squeeze(dim)
    
def safe_unsqueeze(x, dim=0):
    if x is None: 
        return None
    else:
        return x.unsqueeze(dim)    
    
def safe_concat(xs, attr, dim=0):
    if len(xs) == 0: return None
    if getattr(xs[0], attr) is None: return None
    return torch.concat([getattr(i, attr).unsqueeze(dim) for i in xs], dim=dim)

def construct_tuple(x, **kwargs):
    return x(**{k: kwargs[k] if k in kwargs else None for k in x._fields})

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def decode_model_out(model_out, num_actions, enc_type):
    idx1 = num_actions*5+5
    idx2 = num_actions*10+7
    d = dec if enc_type != 0 else lambda x: x
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
        "raw": model_out[0]
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
        self._mean_deques = {}
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

        if name not in self._mean_deques:
            self._mean_deques[name] = collections.deque(maxlen=5)
        self._mean_deques[name].append(x)

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        mean_deques = self._mean_deques
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms (last 5: %.6fms) +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * np.average(mean_deques[k]),
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
        tags = []
        if subname == '_model': tags.append('model')
        m = re.match(r'^v\d+', exp_name)
        if m: tags.append(m[0])
        self.wandb.init(
            project='thinker',
            config=flags,
            entity=os.getenv('WANDB_USER', 'stephen-chung'),
            reinit=True,
            # Restore parameters
            resume="allow",
            id=exp_name,
            name=exp_name,
            tags=tags,
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

def plot_gym_env_out(x, ax=None, title=None, savepath=None):
    if ax is None: _, ax = plt.subplots()
    x = torch.swapaxes(torch.swapaxes(x[0,-3:].detach().cpu(),0,2),0,1).numpy()
    if x.dtype == float: x = np.clip(x, 0, 1)
    if x.dtype == int: x = np.clip(x, 0, 255)
    ax.imshow(x, interpolation='nearest', aspect="auto")
    if title is not None: ax.set_title(title)
    if savepath is not None: 
        plt.savefig(os.path.join(savepath, title+".png"))
        plt.close()