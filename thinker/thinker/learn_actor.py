from collections import deque
import time
import timeit
import os
import numpy as np
import argparse
import ray
import torch
import torch.nn.functional as F

from thinker.core.vtrace import from_importance_weights, VTraceFromLogitsReturns
from thinker.core.file_writer import FileWriter
from thinker.buffer import ActorBuffer, GeneralBuffer
from thinker.buffer import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.net import ActorNet, ModelNet, ActorOut
from thinker.self_play import TrainActorOut
from thinker.env import Environment
import thinker.util as util

def compute_baseline_loss(advantages, masks_ls, c_ls):
    assert len(masks_ls) == len(c_ls)
    loss = 0.  
    for mask, c in zip(masks_ls, c_ls):
        loss = loss + 0.5 * torch.sum((advantages * (1 - mask)) ** 2) * c        
    return loss
    
def compute_policy_gradient_loss(logits_ls, actions_ls, masks_ls, c_ls, advantages):
    assert len(logits_ls) == len(actions_ls) == len(masks_ls) == len(c_ls)
    loss = 0.    
    for logits, actions, masks, c in zip(logits_ls, actions_ls, masks_ls, c_ls):
        cross_entropy = F.nll_loss(
            F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
            target=torch.flatten(actions, 0, 1),
            reduction="none",)
        cross_entropy = cross_entropy.view_as(advantages)
        adv_cross_entropy = cross_entropy * advantages.detach()
        loss = loss + torch.sum(adv_cross_entropy * (1-masks)) * c
    return loss  

def compute_entropy_loss(logits_ls, masks_ls, c_ls):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    loss = 0.
    assert(len(logits_ls) == len(masks_ls) == len(c_ls))
    for logits, masks, c in zip(logits_ls, masks_ls, c_ls):
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        ent = torch.sum(policy * log_policy, dim=-1) #* (1-masks)
        loss = loss + torch.sum(ent) * c 
    return loss

def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions) 
  
def from_logits(
    behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
    discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0, lamb=1.0,):
    """V-trace for softmax policies."""
    assert(len(behavior_logits_ls) == len(target_logits_ls) == len(actions_ls) == len(masks_ls))
    log_rhos = 0.       
    for behavior_logits, target_logits, actions, masks in zip(behavior_logits_ls, 
             target_logits_ls, actions_ls, masks_ls):
        behavior_log_probs = action_log_probs(behavior_logits, actions)        
        target_log_probs = action_log_probs(target_logits, actions)
        log_rho = target_log_probs - behavior_log_probs
        log_rhos = log_rhos + log_rho * (1-masks)
    
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        lamb=lamb
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=None,
        target_action_log_probs=None,
        **vtrace_returns._asdict(),
    )

@ray.remote
class ActorLearner():
    def __init__(self, param_buffer: GeneralBuffer, actor_buffer: ActorBuffer, rank: int, flags: argparse.Namespace):
        self.param_buffer = param_buffer
        self.actor_buffer = actor_buffer
        self.rank = rank
        self.flags = flags
        self._logger = util.logger()
        self.wlogger = util.Wandb(flags, subname='_actor') if flags.use_wandb else None
        self.time = False

        env = Environment(flags, model_wrap=True)
        self.actor_net = ActorNet(obs_shape=env.model_out_shape, 
                                      gym_obs_shape=env.gym_env_out_shape,
                                      num_actions=env.num_actions, 
                                      flags=flags)
        # initialize learning setting

        if not self.flags.disable_cuda and torch.cuda.is_available():
            self._logger.info("Actor-learning: Using CUDA.")
            self.device = torch.device("cuda")
        else:
            self._logger.info("Actor-learning: Not using CUDA.")
            self.device = torch.device("cpu")

        self.step = 0
        self.real_step = 0
        self.tot_eps = 0
        self.last_returns = deque(maxlen=400)
        self.last_im_returns = deque(maxlen=40000)
        
        self.optimizer = torch.optim.Adam(self.actor_net.parameters(),lr=flags.learning_rate)
        if not self.flags.flex_t:
            lr_lambda = lambda epoch: 1 - min(epoch * self.flags.unroll_length * self.flags.batch_size, 
            self.flags.total_steps * self.flags.rec_t) / (self.flags.total_steps * self.flags.rec_t)
        else:
            lr_lambda = lambda epoch: 1 - min(epoch, self.flags.total_steps) / self.flags.total_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.flags.preload_actor and not flags.load_checkpoint:
            checkpoint = torch.load(self.flags.preload_actor, map_location=torch.device('cpu'))
            self.actor_net.set_weights(checkpoint["actor_net_state_dict"])  
            self._logger.info("Loadded actor network from %s" % self.flags.preload_actor)

        if flags.load_checkpoint:
            self.load_checkpoint(os.path.join(flags.load_checkpoint, "ckp_actor.tar"))
            self.flags.savedir = os.path.split(self.flags.load_checkpoint)[0]
            self.flags.xpid = os.path.split(self.flags.load_checkpoint)[-1]    

        self.im_discounting = self.flags.discounting ** (1/self.flags.rec_t)

        # initialize file logs                        
        self.flags.git_revision = util.get_git_revision_hash()
        self.plogger = FileWriter(xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir)

        self.check_point_path = "%s/%s/%s" % (flags.savedir, flags.xpid, "ckp_actor.tar")       

        # set shared buffer's weights
        self.param_buffer.set_data.remote("actor_net", self.actor_net.get_weights())

        # move network and optimizer to process device
        self.actor_net.to(self.device)
        util.optimizer_to(self.optimizer, self.device)

        if self.time: self.timing = util.Timings()

    def learn_data(self):
        timer = timeit.default_timer
        start_step = self.step
        start_time = timer()
        ckp_start_time = int(time.strftime("%M")) // 10
        queue_n = 0
        n = 0
        if self.flags.float16:
            scaler = torch.cuda.amp.GradScaler(init_scale=2**8)
        
        if self.flags.im_cost_anneal: self.anneal_c = 1

        while (self.real_step < self.flags.total_steps):   
            if self.time: self.timing.reset()            
            # get data remotely    
            while (True):
                data = ray.get(self.actor_buffer.read.remote())
                if data is not None: break                
                time.sleep(0.01)
                queue_n += 0.01
            data = ray.get(data)
            if self.time: self.timing.time("get data")

            # start consume data

            # batch and move the data to the process device
            # data is in the form [(train_actor_out_1, initial_actor_state_1), 
            # (train_actor_out_2, initial_actor_state_2) ...]

            train_actor_out = TrainActorOut(*(torch.concat([torch.tensor(x[0][n]) for x in data], dim=1).to(self.device) if data[0][0][n] is not None 
                else None for n in range(len(data[0][0]))))
            initial_actor_state = tuple(torch.concat([x[1][n] for x in data], dim=1).to(self.device) for n in range(len(data[0][1])))
            if self.time: self.timing.time("convert data")

            # compute losses
            if self.flags.float16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):                
                    losses, train_actor_out = self.compute_losses(train_actor_out, initial_actor_state)
            else:
                losses, train_actor_out = self.compute_losses(train_actor_out, initial_actor_state)
            total_loss = losses["total_loss"]
            if self.time: self.timing.time("compute loss")            

            # gradient descent on loss
            self.optimizer.zero_grad()
            if self.flags.float16:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            if self.time: self.timing.time("compute gradient")

            optimize_params = self.optimizer.param_groups[0]['params']
            if self.flags.float16: scaler.unscale_(self.optimizer)
            if self.flags.grad_norm_clipping > 0:                
                total_norm = torch.nn.utils.clip_grad_norm_(optimize_params, self.flags.grad_norm_clipping)
                total_norm = total_norm.detach().cpu().item()
            else:
                total_norm = 0.            
            if self.time: self.timing.time("compute norm")

            if self.flags.float16:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()            
            if self.time: self.timing.time("grad descent")

            if not self.flags.flex_t:
                self.scheduler.step()
            else:
                self.scheduler.last_epoch = self.real_step - 1  # scheduler does not support setting epoch directly
                self.scheduler.step()                    

            if self.flags.im_cost_anneal: self.anneal_c = max(1 - self.real_step / self.flags.total_steps, 0)
            
            # statistic output
            stats = self.compute_stat(train_actor_out, losses, total_norm)       

            # write to log file
            self.plogger.log(stats)
            if self.flags.use_wandb:
                self.wlogger.wandb.log(stats, step=stats['real_step'])

            # print statistics
            if timer() - start_time > 5:
                sps = (self.step - start_step) / (timer() - start_time)                
                print_str =  "Steps %i (%i:%i[%.1f]) @ %.1f SPS. (T_q: %.2f) Eps %i. Return %f (%f). Loss %.2f" % (
                    n, self.real_step, self.step, float(self.step)/float(self.real_step), sps, queue_n, self.tot_eps, 
                    stats["rmean_episode_return"], stats["rmean_im_episode_return"], total_loss)
                print_stats = ["mean_plan_step", "max_rollout_depth", "pg_loss", 
                               "baseline_loss", "im_pg_loss", "im_baseline_loss", 
                               "entropy_loss", "reg_loss", "total_norm"]
                for k in print_stats: print_str += " %s %.2f" % (k, stats[k])
                self._logger.info(print_str)
                start_step = self.step
                start_time = timer()    
                if self.time: print(self.timing.summary()) 
                queue_n = 0
            
            if int(time.strftime("%M")) // 10 != ckp_start_time:
                self.save_checkpoint()
                ckp_start_time = int(time.strftime("%M")) // 10

            if self.time: self.timing.time("misc")  
            del train_actor_out, losses, total_loss, stats, total_norm
            torch.cuda.empty_cache()

            # update shared buffer's weights
            if n % 1 == 0:
                self.param_buffer.set_data.remote("actor_net", self.actor_net.get_weights())
            n += 1
            if self.time: self.timing.time("set weight")  
            
        self.plogger.close()
        return True

    def compute_losses(self, train_actor_out: TrainActorOut, initial_actor_state: tuple):
        # compute loss and then discard the first step in train_actor_out

        T, B = train_actor_out.done.shape
        new_actor_out, unused_state = self.actor_net(train_actor_out, initial_actor_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = new_actor_out.baseline[-1]     

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        train_actor_out = util.tuple_map(train_actor_out, lambda x: x[1:])
        new_actor_out = util.tuple_map(new_actor_out, lambda x: x[:-1])

        rewards = train_actor_out.reward
        # compute advantage w.r.t real rewards     
        discounts = (~train_actor_out.done).float() * self.im_discounting     

        behavior_logits_ls = [train_actor_out.policy_logits, train_actor_out.im_policy_logits, train_actor_out.reset_policy_logits]
        target_logits_ls = [new_actor_out.policy_logits, new_actor_out.im_policy_logits, new_actor_out.reset_policy_logits]
        actions_ls = [train_actor_out.action, train_actor_out.im_action, train_actor_out.reset_action]        
        im_mask = (train_actor_out.cur_t == 0).float()
        real_mask = 1 - im_mask
        zero_mask = torch.zeros_like(im_mask)
        masks_ls = [real_mask, im_mask, im_mask]                
        c_ls = [self.flags.real_cost, self.flags.real_im_cost, self.flags.real_im_cost]
        
        if self.flags.flex_t:
            behavior_logits_ls.append(train_actor_out.term_policy_logits)
            target_logits_ls.append(new_actor_out.term_policy_logits)
            actions_ls.append(train_actor_out.term_action)
            masks_ls.append(zero_mask)
            c_ls.append(self.flags.real_im_cost)

        vtrace_returns = from_logits(
            behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
            discounts=discounts,
            rewards=rewards[:, :, 0],
            values=new_actor_out.baseline[:, :, 0],
            bootstrap_value=bootstrap_value[:, 0],
            lamb=self.flags.lamb
        )    

        pg_loss = compute_policy_gradient_loss(target_logits_ls, actions_ls, masks_ls, c_ls, vtrace_returns.pg_advantages, )          
        baseline_loss = self.flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - new_actor_out.baseline[:, :, 0], 
            masks_ls = [real_mask, im_mask], c_ls = [self.flags.real_cost, self.flags.real_im_cost])

        # compute advantage w.r.t imagainary rewards

        if self.flags.reward_type == 1:
            if self.flags.reward_carry:                
                discounts = (~train_actor_out.done).float() * self.im_discounting 
            else:
                discounts = (~(train_actor_out.cur_t == 0)).float() * self.im_discounting        
            behavior_logits_ls = [train_actor_out.im_policy_logits, train_actor_out.reset_policy_logits]
            target_logits_ls = [new_actor_out.im_policy_logits, new_actor_out.reset_policy_logits]
            actions_ls = [train_actor_out.im_action, train_actor_out.reset_action] 
            masks_ls = [im_mask, im_mask]  
            if not self.flags.im_cost_anneal: 
                c_ls = [self.flags.im_cost, self.flags.im_cost]
            else:
                c_ls = [self.flags.im_cost * self.anneal_c, self.flags.im_cost * self.anneal_c]
            
            if self.flags.flex_t:
                behavior_logits_ls.append(train_actor_out.term_policy_logits)
                target_logits_ls.append(new_actor_out.term_policy_logits)
                actions_ls.append(train_actor_out.term_action)
                masks_ls.append(zero_mask)
                c_ls.append(self.flags.im_cost if not self.flags.im_cost_anneal else self.flags.im_cost * self.anneal_c)
            
            vtrace_returns = from_logits(
                behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
                discounts=discounts,
                rewards=rewards[:, :, 1],
                values=new_actor_out.baseline[:, :, 1],
                bootstrap_value=bootstrap_value[:, 1],
                lamb=self.flags.lamb
            )
            im_pg_loss = compute_policy_gradient_loss(target_logits_ls, actions_ls, masks_ls, c_ls, vtrace_returns.pg_advantages, )   
            im_baseline_loss = self.flags.baseline_cost * compute_baseline_loss(
                vtrace_returns.vs - new_actor_out.baseline[:, :, 1], masks_ls = [zero_mask], c_ls = [
                    self.flags.im_cost if not self.flags.im_cost_anneal else self.flags.im_cost * self.anneal_c])     
        else:
            im_pg_loss = torch.zeros(1, device=self.device)
            im_baseline_loss = torch.zeros(1, device=self.device)
            
        target_logits_ls = [new_actor_out.policy_logits, new_actor_out.im_policy_logits, new_actor_out.reset_policy_logits]
        masks_ls = [real_mask, im_mask, im_mask]    
        im_ent_c = self.flags.im_entropy_cost * (self.flags.real_im_cost + ((
            self.flags.im_cost if not self.flags.im_cost_anneal else self.flags.im_cost * self.anneal_c) if self.flags.reward_type == 1 else 0))
        c_ls = [self.flags.entropy_cost * self.flags.real_cost, im_ent_c, im_ent_c]
        if self.flags.flex_t:
            target_logits_ls.append(new_actor_out.term_policy_logits)
            masks_ls.append(zero_mask)
            c_ls.append(im_ent_c)        
        entropy_loss = compute_entropy_loss(target_logits_ls, masks_ls, c_ls)    

        reg_loss = self.flags.reg_cost * torch.sum(new_actor_out.reg_loss)
        total_loss = pg_loss + baseline_loss + entropy_loss + reg_loss         
            
        if self.flags.reward_type == 1:
            total_loss = total_loss + im_pg_loss + im_baseline_loss      

        losses = {"pg_loss": pg_loss,
                  "im_pg_loss": im_pg_loss,
                  "baseline_loss": baseline_loss,
                  "im_baseline_loss": im_baseline_loss,
                  "entropy_loss": entropy_loss,
                  "reg_loss": reg_loss,
                  "total_loss": total_loss
                  }
        return losses, train_actor_out

    def compute_stat(self, train_actor_out: TrainActorOut, losses: dict, total_norm: float):
        """Update step, real_step and tot_eps; return training stat for printing"""
        episode_returns = train_actor_out.episode_return[train_actor_out.real_done][:, 0]  
        episode_returns = tuple(episode_returns.detach().cpu().numpy())
        self.last_returns.extend(episode_returns)
        rmean_episode_return = np.average(self.last_returns) if len(self.last_returns) > 0 else 0.

        if self.flags.reward_type == 1:
            im_episode_returns = train_actor_out.episode_return[train_actor_out.cur_t==0][:, 1]  
            im_episode_returns = tuple(im_episode_returns.detach().cpu().numpy())
            self.last_im_returns.extend(im_episode_returns)
        rmean_im_episode_return = np.average(self.last_im_returns) if len(self.last_im_returns) > 0 else 0.

        max_rollout_depth = (train_actor_out.max_rollout_depth[train_actor_out.cur_t == 0]).detach().cpu().numpy()
        max_rollout_depth = np.average(max_rollout_depth) if len (max_rollout_depth) > 0 else 0.     
        cur_real_step = torch.sum(train_actor_out.cur_t==0).item()
        mean_plan_step = self.flags.unroll_length * self.flags.batch_size / max(cur_real_step, 1)   

        self.step += self.flags.unroll_length * self.flags.batch_size
        self.real_step += cur_real_step
        self.tot_eps += torch.sum(train_actor_out.real_done).item()

        stats = {"step": self.step,
                    "real_step": self.real_step,
                    "tot_eps": self.tot_eps,
                    "rmean_episode_return": rmean_episode_return,
                    "rmean_im_episode_return": rmean_im_episode_return, 
                    "episode_returns": episode_returns,                     
                    "cur_real_step": cur_real_step,
                    "mean_plan_step": mean_plan_step,
                    "max_rollout_depth": max_rollout_depth,
                    "total_norm": total_norm
                    }

        for k, v in losses.items(): stats[k] = v.item()  
        return stats

    def save_checkpoint(self):
        self._logger.info("Saving actor checkpoint to %s" % self.check_point_path)
        torch.save(
            { "step": self.step,
              "real_step": self.real_step,
              "tot_eps": self.tot_eps,
              "last_returns": self.last_returns,
              "last_im_returns": self.last_im_returns,
              "actor_net_optimizer_state_dict": self.optimizer.state_dict(),
              "actor_net_scheduler_state_dict": self.scheduler.state_dict(),
              "actor_net_state_dict": self.actor_net.state_dict(),
              "flags":  vars(self.flags)
            },
            self.check_point_path,
        )
    
    def load_checkpoint(self, check_point_path: str):
        train_checkpoint = torch.load(check_point_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.last_returns = train_checkpoint["last_returns"]
        self.last_im_returns = train_checkpoint["last_im_returns"]
        self.optimizer.load_state_dict(train_checkpoint["actor_net_optimizer_state_dict"])         
        self.scheduler.load_state_dict(train_checkpoint["actor_net_scheduler_state_dict"])       
        self.actor_net.set_weights(train_checkpoint["actor_net_state_dict"])        
        self._logger.info("Loaded actor checkpoint from %s" % check_point_path)   
