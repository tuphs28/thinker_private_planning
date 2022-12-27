import collections
import time
import timeit
import argparse
import subprocess
import os 
import torch

def parse(args=None):
    parser = argparse.ArgumentParser(description="Thinker v1")

    parser.add_argument("--env", type=str, default="cSokoban-v0",
                        help="Gym environment.")
    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")
    parser.add_argument("--savedir", default="~/RS/thinker/logs/thinker",
                        help="Root dir where experiment data will be saved.")

    # Preload settings.
    parser.add_argument("--load_checkpoint", default="",
                        help="Load checkpoint directory.")    
    parser.add_argument("--preload_actor", default="",
                        help="File location of the preload actor network.")                        
    parser.add_argument("--preload_model", default="~/RS/thinker/models/model_1.tar",
                        help="File location of the preload model network.")
    parser.add_argument("--employ_model", default="",
                        help="Use another fixed model for the planning agent")                        
    parser.add_argument("--employ_model_rnn",  action="store_true",
                        help="Whether to use ConvLSTM in the employed model (only support perfect model).")                        


    # Actor Training settings.            
    parser.add_argument("--policy_type", default=0, type=int, 
                        help="Policy used for self-play worker; 0 for actor net, 1 for model policy, 2 for 1-step greedy") 
    parser.add_argument("--disable_train_actor", action="store_false", dest="train_actor",
                        help="Disable training of actor.")   
    parser.add_argument("--num_actors", default=48, type=int, 
                        help="Number of actors (default: 48).")
    parser.add_argument("--actor_parallel_n", default=1, type=int, 
                        help="Number of parallel env. per actor")
    parser.add_argument("--total_steps", default=50000000, type=int, 
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=32, type=int, 
                        help="Actor learner batch size.")
    parser.add_argument("--unroll_length", default=200, type=int, 
                        help="The unroll length (time dimension).")
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")

    # Model Training settings. 
    parser.add_argument("--train_model", action="store_true",
                        help="Enable training of model.")
    parser.add_argument("--model_batch_size", default=128, type=int, 
                        help="Model learner batch size.")   
    parser.add_argument("--model_batch_mode", action="store_true",
                        help="Whether to use the full rollout from model buffer in training.")                                            
    parser.add_argument("--model_unroll_length", default=200, type=int, 
                        help="Number of transition per unroll in model buffer.")
    parser.add_argument("--model_k_step_return", default=5, type=int, 
                        help="Number of recurrent step when training the model.")    
    parser.add_argument("--priority_alpha", default=0.6, type=float,
                        help="Alpha used to compute the priority from model buffer.")
    parser.add_argument("--priority_beta", default=0.4, type=float,
                        help="Initial beta used to compute the priority from model buffer.")
    parser.add_argument("--model_buffer_n", default=200000, type=int, 
                        help="Maximum number of transition in model buffer.") 
    parser.add_argument("--model_warm_up_n", default=400000, type=int, 
                        help="Number of transition accumulated before model start learning.")                        
    parser.add_argument("--test_policy_type", default=1, type=int, 
                        help="Policy used for testing model; 0 for actor net, 1 for model policy, 2 for 1-step greedy")                         
    parser.add_argument("--model_min_step_per_transition", default=-1, type=int, 
                        help="Minimum number of model learning step on one transition")                         
    parser.add_argument("--model_max_step_per_transition", default=-1, type=int, 
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
    
    # Model architecure settings
    parser.add_argument("--model_type_nn", default=0,
                        type=float, help="Model type.")        
    parser.add_argument("--model_rnn", action="store_true",
                        help="Whether to use ConvLSTM in model (only support perfect model).")    
    
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
    parser.add_argument("--discounting", default=0.97,
                        type=float, help="Discounting factor.")
    parser.add_argument("--lamb", default=1,
                        type=float, help="Lambda when computing trace.")
    parser.add_argument("--reward_clipping", default=10, type=int, 
                       help="Reward clipping.")
    
    # Model loss settings
    parser.add_argument("--model_logits_loss_cost", default=0.05, type=float,
                       help="Multipler to policy logit loss when training the model.")                            
    parser.add_argument("--model_vs_loss_cost", default=1, type=float,
                       help="Multipler to policy vs loss when training the model.")                           
    
    # Model wrapper settings
    parser.add_argument("--reward_type", default=1, type=int, 
                        help="Reward type")   
    parser.add_argument("--reset_m", default=-1, type=int,
                        help="Auto reset after passing m node since an unexpanded noded")     
    parser.add_argument("--disable_perfect_model", action="store_false", dest="perfect_model",
                        help="Whether to use perfect model.")          
    parser.add_argument("--rec_t", default=40, type=int, 
                        help="Number of planning steps.")
    parser.add_argument("--flex_t", action="store_true",
                        help="Whether to enable flexible planning steps.") 
    parser.add_argument("--flex_t_cost", default=1e-5,
                        type=float, help="Cost of planning step (only enabled when flex_t == True).")               
    parser.add_argument("--flex_t_term_b", default=0.,
                        type=float, help="Bias added to the logit of term action.")      
    parser.add_argument("--no_mem", action="store_true",
                        help="Whether to erase all memories after each real action.")   
    parser.add_argument("--disable_tree_carry", action="store_false", dest="tree_carry",
                        help="Whether to carry over the tree.")   
    parser.add_argument("--thres_carry", action="store_true",
                        help="Whether to carry threshold over.")   
    parser.add_argument("--reward_carry", action="store_true",
                        help="Whether to carry planning reward over.")      
    parser.add_argument("--thres_discounting", default=0.9,
                        type=float, help="Threshold discounting factor.")   

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
    
    fs = ["load_checkpoint", "savedir", "preload_model", "preload_actor", "employ_model"]    
    for f in fs:
        path = getattr(flags, f)
        if path: setattr(flags, f, os.path.expanduser(path))
            
    if flags.load_checkpoint:
        check_point_path = os.path.join(flags.load_checkpoint, "ckp_actor.tar")
        train_checkpoint = torch.load(check_point_path, torch.device("cpu"))
        flags_ = train_checkpoint["flags"]
        for k, v in flags_.items(): 
            if k != "load_checkpoint": setattr(flags, k, v)

    if flags.xpid is None:
        flags.xpid = "thinker-%s" % time.strftime("%Y%m%d-%H%M%S")

    if flags.model_rnn:
        assert flags.model_batch_mode, "rnn model can only be equipped with batch model mode"

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
