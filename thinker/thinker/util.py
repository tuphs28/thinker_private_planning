import collections
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

    parser.add_argument("--load_checkpoint", default="",
                        help="Load checkpoint directory.")    
    parser.add_argument("--savedir", default="~/RS/thinker/logs/thinker",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--preload_model", default="~/RS/thinker/models/model_1.tar",
                        help="File location of the preload model network.")

    # Training settings.        
    parser.add_argument("--num_actors", default=48, type=int, metavar="N",
                        help="Number of actors (default: 48).")
    parser.add_argument("--total_steps", default=50000000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=32, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--unroll_length", default=200, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")

    # Architecture settings
    parser.add_argument("--tran_dim", default=96, type=int, metavar="N",
                        help="Size of transformer hidden dim.")
    parser.add_argument("--tran_mem_n", default=40, type=int, metavar="N",
                        help="Size of transformer memory.")
    parser.add_argument("--tran_layer_n", default=3, type=int, metavar="N",
                        help="Number of transformer layer.")
    parser.add_argument("--tran_t", default=1, type=int, metavar="T",
                        help="Number of recurrent step for transformer.")   
    parser.add_argument("--tran_lstm_no_attn", action="store_true",
                        help="Whether to disable attention in LSTM-transformer.")
    parser.add_argument("--tran_attn_b", default=5.,
                        type=float, help="Bias attention for current position.")        
    
    # Loss settings.
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
    parser.add_argument("--lamb", default=1.,
                        type=float, help="Lambda when computing trace.")
    parser.add_argument("--reward_clipping", default=10, type=int, 
                        metavar="N", help="Reward clipping.")
    
    # Model settings
    parser.add_argument("--reward_type", default=1, type=int, metavar="N",
                        help="Reward type")   
    parser.add_argument("--reset_m", default=-1, type=int, metavar="N",
                        help="Auto reset after passing m node since an unexpanded noded")    
    parser.add_argument("--model_type_nn", default=0,
                        type=float, help="Model type.")     
    parser.add_argument("--disable_perfect_model", action="store_false", dest="perfect_model",
                        help="Whether to use perfect model.")          
    parser.add_argument("--rec_t", default=40, type=int, metavar="N",
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
                        type=float, metavar="LR", help="Learning rate.")
    parser.add_argument("--grad_norm_clipping", default=600, type=float,
                        help="Global gradient norm clip.")

    if args is None:
        flags = parser.parse_args()  
    else:
        flags = parser.parse_args(args)  
    
    fs = ["load_checkpoint", "savedir", "preload_model"]    
    for f in fs:
        path = getattr(flags, f)
        if path: setattr(flags, f, os.path.expanduser(path))

    return flags

def tuple_map(x, f):
    return type(x)(*(f(y) if y is not None else None for y in x))

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
