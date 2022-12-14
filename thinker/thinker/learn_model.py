import os
import numpy as np
import argparse
import time
import timeit
import ray
import torch
import torch.nn.functional as F
from thinker.buffer import ParamBuffer, ModelBuffer
from thinker.net import ModelNet
from thinker.self_play import TrainModelOut
from thinker.env import Environment
import thinker.util as util

def compute_cross_entropy_loss(logits, target_logits, mask, is_weights):
    target_policy = F.softmax(target_logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(target_policy * log_policy * (~mask).float().unsqueeze(-1), dim=(0,2))
    loss = is_weights * loss
    return torch.sum(loss)

@ray.remote(num_cpus=1, num_gpus=0.5)
class ModelLearner():
    def __init__(self, param_buffer: ParamBuffer, model_buffer: ModelBuffer, rank: int, flags: argparse.Namespace): 
        self.param_buffer = param_buffer
        self.model_buffer = model_buffer
        self.rank = rank
        self.flags = flags

        env = Environment(flags)        
        self.model_net = ModelNet(obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags)
        self.model_net.train(True)

        # initialize learning setting

        if not self.flags.disable_cuda and torch.cuda.is_available():
            print("Model-learning: Using CUDA.")
            self.device = torch.device("cuda")
        else:
            print("Model-learning: Not using CUDA.")
            self.device = torch.device("cpu")

        self.step = 0
        self.real_step = 0
        
        self.optimizer = torch.optim.Adam(self.model_net.parameters(),lr=flags.model_learning_rate)
        lr_lambda = lambda epoch: 1 - min(epoch, self.flags.total_steps) / self.flags.total_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.flags.preload_model and not flags.load_checkpoint:
            checkpoint = torch.load(self.flags.preload_model, map_location=torch.device('cpu'))
            self.model_net.set_weights(checkpoint["model_state_dict"])  
            print("Loadded model network from %s" % self.flags.preload_model)

        if flags.load_checkpoint:
            self.load_checkpoint(os.path.join(flags.load_checkpoint, "ckp_model.tar"))
            self.flags.savedir = os.path.split(os.path.split(self.flags.load_checkpoint)[0])[0]
            self.flags.xpid = os.path.split(os.path.split(self.flags.load_checkpoint)[0])[-1]    

        self.check_point_path = "%s/%s/%s" % (flags.savedir, flags.xpid, "ckp_model.tar")

        # set shared buffer's weights
        self.param_buffer.set_weights.remote("model_net", self.model_net.get_weights())

        # move network and optimizer to process device
        self.model_net.to(self.device)
        util.optimizer_to(self.optimizer, self.device)

    def learn_data(self):
        timer = timeit.default_timer
        start_step = self.step
        start_time = timer()
        ckp_start_time = int(time.strftime("%M")) // 10
        n = 0

        max_diff = self.flags.model_unroll_length * self.flags.num_actors * self.flags.actor_parallel_n
        # in case the actor learner stops before model learner

        while (self.real_step < self.flags.total_steps - max_diff):                    
            c = min(self.real_step, self.flags.total_steps) / self.flags.total_steps
            beta = self.flags.priority_beta * (1 - c) + 1. * c

            # get data remotely    
            while (True):
                data = ray.get(self.model_buffer.read.remote(beta))
                if data is not None: break            
                time.sleep(0.01)

            # start consume data
            train_model_out, is_weights, inds, self.real_step = data

            # move the data to the process device
            train_model_out = util.tuple_map(train_model_out, lambda x:x.to(self.device))
            is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
                        
            # compute losses
            losses, priorities = self.compute_losses(train_model_out, is_weights)
            total_loss = losses["total_loss"]
            self.model_buffer.update_priority.remote(inds, priorities)
            
            # gradient descent on loss
            self.optimizer.zero_grad()
            total_loss.backward()
            optimize_params = self.optimizer.param_groups[0]['params']
            if self.flags.model_grad_norm_clipping > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(optimize_params, self.flags.model_grad_norm_clipping)
            else:
                total_norm = 0.
            
            self.optimizer.step()
            self.scheduler.last_epoch = self.real_step - 1  # scheduler does not support setting epoch directly
            self.scheduler.step()                 
            
            self.step += self.flags.model_batch_size
            # print statistics
            if timer() - start_time > 5:
                sps = (self.step - start_step) / (timer() - start_time)                
                print_str =  "Steps %i (%i:%i) @ %.1f SPS. Loss %.2f" % (
                    n, self.real_step, self.step, sps, total_loss)
                print_stats = ["vs_loss", "logits_loss", "rs_loss"]
                for k in print_stats: 
                    if losses[k] is not None:
                        print_str += " %s %.2f" % (k, losses[k].item())
                print(print_str)
                start_step = self.step
                start_time = timer()      
            
            if int(time.strftime("%M")) // 10 != ckp_start_time:
                self.save_checkpoint()
                ckp_start_time = int(time.strftime("%M")) // 10
            
            # update shared buffer's weights
            if n % 1 == 0:
                self.param_buffer.set_weights.remote("model_net", self.model_net.get_weights())
            n += 1
        return True

    def compute_losses(self, train_model_out: TrainModelOut, is_weights: torch.Tensor):
        k, b = self.flags.model_k_step_return, self.flags.model_batch_size
        if self.flags.perfect_model:            
            _, vs, logits, _ = self.model_net(
                x=train_model_out.gym_env_out.reshape((-1,) + train_model_out.gym_env_out.shape[2:]),
                actions=train_model_out.action.reshape(1, -1),
                one_hot=False)
            vs = vs.reshape(k+1, b)
            logits = logits.reshape(k+1, b, -1)
        else:        
            rs, vs, logits, _ = self.model_net(
                x=train_model_out.gym_env_out[0], 
                actions=train_model_out.action,
                one_hot=False)

        logits = logits[:-1]
        target_rewards = train_model_out.reward[1:]
        target_logits = train_model_out.policy_logits[1:]

        target_vs = []
        target_v = self.model_net(train_model_out.gym_env_out[-1], train_model_out.action[[-1]])[1][0].detach()        
        for t in range(k, 0, -1):
            new_target_v = train_model_out.reward[t] + self.flags.discounting * (
                target_v * (~train_model_out.done[t]).float())
            target_vs.append(new_target_v.unsqueeze(0))
            target_v = new_target_v

        target_vs.reverse()
        target_vs = torch.concat(target_vs, dim=0)

        # if done on step j, r_{j}, v_{j-1}, a_{j-1} has the last valid loss 
        # rs is stored in the form of r_{t+1}, ..., r_{t+k}
        # vs is stored in the form of v_{t}, ..., v_{t+k-1}
        # logits is stored in the form of a{t}, ..., a_{t+k-1}

        done_masks = []
        done = torch.zeros(b).bool().to(self.device)
        for t in range(k):
            if t > 0: done = torch.logical_or(done, train_model_out.done[t])
            done_masks.append(done.unsqueeze(0))

        done_masks = torch.concat(done_masks, dim=0)
        
        # compute final loss

        #huberloss = torch.nn.HuberLoss(reduction='none', delta=1.0)    
        #rs_loss = torch.sum(huberloss(rs, target_rewards.detach()) * (~done_masks).float())
        if self.flags.perfect_model:
            rs_loss =  None
        else:
            rs_loss = torch.sum(((rs - target_rewards) ** 2) * (~done_masks).float(), dim=0)
            rs_loss = rs_loss * is_weights
            rs_loss = torch.sum(rs_loss)
        #vs_loss = torch.sum(huberloss(vs[:-1], target_vs.detach()) * (~done_masks).float())
        vs_loss = torch.sum(((vs[:-1] - target_vs) ** 2) * (~done_masks).float(), dim=0)
        vs_loss = vs_loss * is_weights
        vs_loss = torch.sum(vs_loss)

        logits_loss = self.flags.model_logits_loss_cost * compute_cross_entropy_loss(
            logits, target_logits.detach(), done_masks, is_weights)        

        total_loss = vs_loss + logits_loss
        if rs_loss is not None: total_loss = total_loss + rs_loss

        losses = {"total_loss": total_loss,
                   "vs_loss": vs_loss,
                   "logits_loss": logits_loss,
                   "rs_loss": rs_loss }
        priorities = ((vs[0] - target_vs[0]) ** 2).detach().cpu().numpy()

        return losses, priorities

    def save_checkpoint(self):
        print("Saving model checkpoint to %s" % self.check_point_path)
        torch.save(
            { "step": self.step,
              "real_step": self.real_step,
              "model_net_optimizer_state_dict": self.optimizer.state_dict(),
              "model_net_scheduler_state_dict": self.scheduler.state_dict(),
              "model_net_state_dict": self.model_net.state_dict(),
              "flags":  vars(self.flags)
            },
            self.check_point_path,
        )

    def load_checkpoint(self, check_point_path: str):
        train_checkpoint = torch.load(check_point_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.optimizer.load_state_dict(train_checkpoint["model_net_optimizer_state_dict"])         
        self.scheduler.load_state_dict(train_checkpoint["model_net_scheduler_state_dict"])       
        self.model_net.set_weights(train_checkpoint["model_net_state_dict"])        
        print("Loaded model checkpoint from %s" % check_point_path)   