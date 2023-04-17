from collections import deque
import time
import timeit
import os
import numpy as np
import argparse
import traceback
import ray
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from thinker.core.vtrace import from_importance_weights, VTraceFromLogitsReturns, FifoBuffer
from thinker.core.file_writer import FileWriter
from thinker.buffer import ActorBuffer, GeneralBuffer
from thinker.buffer import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.net import ActorNet, ModelNet, ActorOut
from thinker.self_play import TrainActorOut
from thinker.env import Environment
import thinker.util as util


def compute_baseline_loss(new_actor_out, 
                          ind, 
                          target_baseline, 
                          actor_net,                           
                          c, 
                          enc_type):    
    target_baseline = target_baseline.detach()
    if enc_type == 0:
        baseline = new_actor_out.baseline[:, :, ind]
        advantages = baseline - target_baseline
        loss = torch.sum(advantages ** 2) * c    
    elif enc_type == 1:
        rv_tran = actor_net.rv_tran
        baseline_enc = new_actor_out.baseline_enc[:, :, ind]
        target_baseline_enc = rv_tran.encode(target_baseline)
        advantages = baseline_enc - target_baseline_enc
        loss = torch.sum(advantages ** 2) * c
    elif enc_type in [2, 3]:
        rv_tran = actor_net.rv_tran
        baseline_enc = new_actor_out.baseline_enc[:, :, ind]
        target_baseline_enc = rv_tran.encode(target_baseline)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")(
            input = torch.flatten(baseline_enc, 0, 1),
            target = torch.flatten(target_baseline_enc, 0, 1)
        ) * c
    elif enc_type == 4:     
        baseline_enc = new_actor_out.baseline_enc[:, :, ind]
        baseline_scale = actor_net.baseline_scale[ind]
        advantages = baseline_enc - target_baseline / baseline_scale
        loss = torch.sum(advantages ** 2) * c  
    return loss

def compute_policy_gradient_loss(logits_ls, actions_ls, masks_ls, c_ls, advantages):
    assert len(logits_ls) == len(actions_ls) == len(masks_ls) == len(c_ls)
    loss = 0.    
    for logits, actions, masks, c in zip(logits_ls, actions_ls, masks_ls, c_ls):
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")(
            input = torch.flatten(logits, 0, 1),
            target = torch.flatten(actions, 0, 1)
            )
        cross_entropy = cross_entropy.view_as(advantages)
        adv_cross_entropy = cross_entropy * advantages.detach()
        loss = loss + torch.sum(adv_cross_entropy * (1-masks)) * c
    return loss  

def compute_entropy_loss(logits_ls, masks_ls, c_ls, mask_ent):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    loss = 0.
    assert(len(logits_ls) == len(masks_ls) == len(c_ls))
    for logits, masks, c in zip(logits_ls, masks_ls, c_ls):
        logits = torch.flatten(logits, 0, 1)
        ent = -torch.nn.CrossEntropyLoss(reduction="none")(
            input = logits,
            target = F.softmax(logits, dim=-1))
        ent = ent.view_as(masks)  
        if mask_ent: ent = ent * (1-masks)
        loss = loss + torch.sum(ent) * c 
    return loss

def action_log_probs(policy_logits, actions):
    return -torch.nn.CrossEntropyLoss(reduction="none")(
            input = torch.flatten(policy_logits, 0, 1),
            target = torch.flatten(actions, 0, 1)).view_as(actions) 
            
def from_logits(
    behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
    discounts, rewards, values, values_enc, rv_tran, enc_type, bootstrap_value, flags,
    clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, lamb=1.0, norm_stat=None):
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
        values_enc=values_enc, 
        rv_tran=rv_tran,
        enc_type=enc_type,
        bootstrap_value=bootstrap_value,
        flags=flags,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        lamb=lamb,
        norm_stat=norm_stat
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=None,
        target_action_log_probs=None,
        **vtrace_returns._asdict(),
    )

@torch.no_grad()
def update_b_norm_stat(b_norm_stat, target_baseline):
    if b_norm_stat is None:
        buffer = FifoBuffer(100000, device=target_baseline.device)
    else:
        buffer = b_norm_stat[2]
    buffer.push(target_baseline)
    lq = buffer.get_percentile(0.05)
    uq = buffer.get_percentile(0.95)
    return (lq, uq, buffer)

@torch.no_grad()
def update_baseline_scale(b_norm_stat, actor_net, n, norm_b):
    old_scale = torch.clone(actor_net.baseline_scale[n])
    new_scale = torch.clamp(b_norm_stat[1] - b_norm_stat[0], min=norm_b, max=None)
    new_scale = old_scale * torch.clamp(new_scale / old_scale, min=0.9, max=1.1)
    actor_net.baseline_scale[n] = new_scale
    actor_net.baseline.weight.data[n, :] *= (old_scale / new_scale)
    actor_net.baseline.bias.data[n] *= (old_scale / new_scale)

@ray.remote
class ActorLearner():
    def __init__(self, buffers:dict, rank: int, flags: argparse.Namespace):
        self.param_buffer = buffers["actor_param"]
        self.actor_buffer = buffers["actor"]
        self.rank = rank
        self.flags = flags
        self._logger = util.logger()
        self.time = flags.profile

        env = Environment(flags, model_wrap=True, env_n=1)
        self.actor_net = ActorNet(obs_shape=env.model_out_shape, 
                                      gym_obs_shape=env.gym_env_out_shape,
                                      num_actions=env.num_actions, 
                                      flags=flags)
        env.close()
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
        self.last_cur_returns = deque(maxlen=400)
        
        self.optimizer = torch.optim.Adam(self.actor_net.parameters(),lr=flags.learning_rate)
        lr_lambda = lambda epoch: 1 - min(epoch * self.flags.unroll_length * self.flags.batch_size, 
        self.flags.total_steps * self.flags.rec_t) / (self.flags.total_steps * self.flags.rec_t)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.flags.preload_actor and not flags.load_checkpoint:
            checkpoint = torch.load(self.flags.preload_actor, map_location=torch.device('cpu'))
            self.actor_net.set_weights(checkpoint["actor_net_state_dict"])  
            self._logger.info("Loadded actor network from %s" % self.flags.preload_actor)
            if "actor_net_optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["actor_net_optimizer_state_dict"])  
                self._logger.info("Loadded actor network's optimizer from %s" % self.flags.preload_actor)

        if flags.load_checkpoint:
            self.load_checkpoint(os.path.join(flags.load_checkpoint, "ckp_actor.tar"))
            self.flags.savedir = os.path.split(self.flags.load_checkpoint)[0]
            self.flags.xpid = os.path.split(self.flags.load_checkpoint)[-1]    

        self.im_discounting = self.flags.discounting ** (1/self.flags.rec_t)

        # initialize file logs                        
        self.flags.git_revision = util.get_git_revision_hash()
        self.plogger = FileWriter(xpid=flags.xpid, xp_args=flags.__dict__, 
                                  rootdir=flags.savedir, overwrite=not self.flags.load_checkpoint)

        self.check_point_path = "%s/%s/%s" % (flags.savedir, flags.xpid, "ckp_actor.tar")       

        # set shared buffer's weights
        self.param_buffer.set_data.remote("actor_net", self.actor_net.get_weights())

        # move network and optimizer to process device
        self.actor_net.to(self.device)
        util.optimizer_to(self.optimizer, self.device)
        if self.time: self.timing = util.Timings()        

        self.norm_stat = None
        self.im_norm_stat = None
        self.cur_norm_stat = None

        self.b_norm_stat = None
        self.b_im_norm_stat = None
        self.b_cur_norm_stat = None

    def learn_data(self):
        try:
            timer = timeit.default_timer     
            start_time = timer()
            sps = 0
            sps_buffer = [(self.step, start_time)] * 36
            sps_buffer_n = 0
            sps_start_time, sps_start_step = start_time, self.step
            ckp_start_time = int(time.strftime("%M")) // 10
            queue_n = 0
            n = 0
            if self.flags.float16:
                scaler = torch.cuda.amp.GradScaler(init_scale=2**8)
            
            self.anneal_c = 1

            data_ptr = self.actor_buffer.read.remote()
            while (self.real_step < self.flags.total_steps):   
                if self.time: self.timing.reset()            
                # get data remotely    
                while (True):
                    data_ptr = self.actor_buffer.read.remote()
                    data = ray.get(data_ptr)                    
                    if data is not None: break                
                    time.sleep(0.001)
                    queue_n += 0.001
                if self.time: self.timing.time("get_data")
                # start consume data

                train_actor_out, initial_actor_state  = data
                train_actor_out = util.tuple_map(train_actor_out, lambda x:torch.tensor(x, device=self.device))
                initial_actor_state = util.tuple_map(initial_actor_state, lambda x:torch.tensor(x, device=self.device))
                actor_id = train_actor_out.id                
                if self.time: self.timing.time("convert_data")

                if self.real_step < self.flags.actor_warm_up_n:
                    stats = self.compute_stat(train_actor_out, None, None, actor_id)   
                    self._logger.info("Preloading: %d/%d" % (self.real_step, self.flags.actor_warm_up_n))
                    time.sleep(5)
                    continue                           
                
                # compute losses
                if self.flags.float16:
                    with torch.cuda.amp.autocast():
                        losses, train_actor_out = self.compute_losses(train_actor_out, initial_actor_state)
                else:
                    losses, train_actor_out = self.compute_losses(train_actor_out, initial_actor_state)
                total_loss = losses["total_loss"]
                if self.time: self.timing.time("compute loss")    

                #with profile(activities=[
                #    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
                #    with record_function("compute_grad"):

                # gradient descent on loss
                self.optimizer.zero_grad()
                if self.flags.float16:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                if self.time: self.timing.time("compute gradient")

                #print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

                optimize_params = self.optimizer.param_groups[0]['params']
                if self.flags.float16: scaler.unscale_(self.optimizer)
                if self.flags.grad_norm_clipping > 0:                
                    total_norm = torch.nn.utils.clip_grad_norm_(optimize_params, self.flags.grad_norm_clipping)
                    total_norm = total_norm.detach().cpu().item()
                else:
                    total_norm = util.compute_grad_norm(optimize_params)     
                if self.time: self.timing.time("compute norm")

                if self.flags.float16:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()            
                if self.time: self.timing.time("grad descent")

                self.scheduler.step()         

                if self.flags.critic_enc_type == 4:
                    # update baseline_scale
                    update_baseline_scale(self.b_norm_stat, self.actor_net, 0, self.flags.critic_norm_b)
                    if self.flags.im_cost > 0.:
                        update_baseline_scale(self.b_im_norm_stat, self.actor_net, 1, self.flags.critic_norm_b)
                    if self.flags.cur_cost > 0. and self.real_step > self.flags.cur_warm_up_n:
                        update_baseline_scale(self.b_cur_norm_stat, self.actor_net, 2, self.flags.critic_norm_b)

                self.anneal_c = max(1 - self.real_step / (
                        self.flags.total_steps), 0)
                
                # statistic output
                stats = self.compute_stat(train_actor_out, losses, total_norm, actor_id) 
                stats["sps"] = sps                  

                # write to log file
                self.plogger.log(stats)        

                # print statistics
                if timer() - start_time > 5:                    
                    sps_buffer[sps_buffer_n] = (self.step, timer())
                    sps_buffer_n = (sps_buffer_n + 1) % len(sps_buffer)
                    sps = ((sps_buffer[sps_buffer_n-1][0] - sps_buffer[sps_buffer_n][0]) / 
                           (sps_buffer[sps_buffer_n-1][1] - sps_buffer[sps_buffer_n][1]))                    
                    tot_sps = (self.step - sps_start_step) / (timer() - sps_start_time) 
                    print_str =  "[%s] Steps %i @ %.1f SPS (%.1f). (T_q: %.2f) Eps %i. Ret %f (%f/%f). Loss %.2f" % (
                        self.flags.xpid, self.real_step, sps, tot_sps, 
                        queue_n, self.tot_eps, stats["rmean_episode_return"], 
                        stats["rmean_im_episode_return"],  stats["rmean_cur_episode_return"], total_loss)
                    print_stats = ["max_rollout_depth", "entropy_loss", "reg_loss", "total_norm"]                    
                    for k in print_stats: print_str += " %s %.2f" % (k, stats[k])
                    if self.flags.return_norm_type != -1:
                        print_str += " norm_diff (%.4f/%.4f/%.4f)" % (
                            stats["norm_diff"], stats["im_norm_diff"], stats["cur_norm_diff"])
                    if self.flags.critic_enc_type == 4:
                        print_str += " b_norm_diff (%.4f/%.4f/%.4f)" % (
                            stats["b_norm_diff"], stats["b_im_norm_diff"], stats["b_cur_norm_diff"])

                    self._logger.info(print_str)
                    start_time = timer()                    
                    queue_n = 0
                    if self.time: print(self.timing.summary()) 
                
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
            
            self.close(0)  
            return True        
        
        except Exception as e:
            self._logger.error(f"Exception detected in learn_actor: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            self.close(0)  
            return True 

    def close(self, exit_code=0):
        self.plogger.close()

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

        if self.flags.only_cur and self.flags.cur_cost > 0.:
            rewards[:, :, 0] = rewards[:, :, 2]

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

        vtrace_returns = from_logits(
            behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
            discounts=discounts,
            rewards=rewards[:, :, 0],
            values=new_actor_out.baseline[:, :, 0],
            values_enc=new_actor_out.baseline_enc[:, :, 0] if new_actor_out.baseline_enc is not None else None,
            rv_tran=self.actor_net.rv_tran,
            enc_type=self.flags.critic_enc_type,                
            bootstrap_value=bootstrap_value[:, 0],
            flags=self.flags,
            lamb=self.flags.lamb,
            norm_stat=self.norm_stat
        )    
        if self.flags.critic_enc_type == 4: 
            self.b_norm_stat = update_b_norm_stat(
                b_norm_stat=self.b_norm_stat,
                target_baseline=vtrace_returns.vs,      
            )
        self.norm_stat = vtrace_returns.norm_stat
        advs = vtrace_returns.pg_advantages        
        pg_loss = compute_policy_gradient_loss(target_logits_ls, actions_ls, masks_ls, c_ls, advs)          
        baseline_loss = self.flags.baseline_cost * compute_baseline_loss(
            new_actor_out=new_actor_out,
            ind=0,            
            target_baseline=vtrace_returns.vs,
            actor_net=self.actor_net,
            c=self.flags.real_cost,
            enc_type=self.flags.critic_enc_type,
        )
            
        # compute advantage w.r.t curiosity rewards
        if self.flags.cur_cost > 0. and self.real_step > self.flags.cur_warm_up_n:
            discounts = (~train_actor_out.done).float() * self.im_discounting   
            behavior_logits_ls = [train_actor_out.policy_logits, train_actor_out.im_policy_logits, train_actor_out.reset_policy_logits]
            target_logits_ls = [new_actor_out.policy_logits, new_actor_out.im_policy_logits, new_actor_out.reset_policy_logits]
            actions_ls = [train_actor_out.action, train_actor_out.im_action, train_actor_out.reset_action] 
            masks_ls = [real_mask, im_mask, im_mask]    
            c_ls = [self.flags.cur_cost * self.anneal_c if self.flags.cur_cost_anneal else self.flags.cur_cost,
                    self.flags.cur_im_cost * self.anneal_c if self.flags.cur_cost_anneal else self.flags.cur_im_cost,
                    self.flags.cur_im_cost * self.anneal_c if self.flags.cur_cost_anneal else self.flags.cur_im_cost]

            vtrace_returns = from_logits(
                behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
                discounts=discounts,
                rewards=rewards[:, :, 2],
                values=new_actor_out.baseline[:, :, 2],
                values_enc=new_actor_out.baseline_enc[:, :, 2] if new_actor_out.baseline_enc is not None else None,
                rv_tran=self.actor_net.rv_tran,
                enc_type=self.flags.critic_enc_type,                
                bootstrap_value=bootstrap_value[:, 2],
                flags=self.flags,
                lamb=self.flags.lamb,
                norm_stat=self.cur_norm_stat
            )    
            if self.flags.critic_enc_type == 4: 
                self.b_cur_norm_stat = update_b_norm_stat(
                    b_norm_stat=self.b_cur_norm_stat,
                    target_baseline=vtrace_returns.vs,      
                )
            self.cur_norm_stat = vtrace_returns.norm_stat
            advs = vtrace_returns.pg_advantages            
            
            cur_pg_loss = compute_policy_gradient_loss(target_logits_ls, actions_ls, masks_ls, c_ls, advs)            
            cur_baseline_loss = self.flags.baseline_cost * compute_baseline_loss(
                new_actor_out=new_actor_out,
                ind=2,            
                target_baseline=vtrace_returns.vs,
                actor_net=self.actor_net,
                c=self.flags.cur_cost if not self.flags.cur_cost_anneal else self.flags.cur_cost * self.anneal_c,
                enc_type=self.flags.critic_enc_type,
            )            
        else:
            cur_pg_loss = torch.zeros(1, device=self.device)
            cur_baseline_loss = torch.zeros(1, device=self.device)

        # compute advantage w.r.t imagainary rewards

        if self.flags.im_cost > 0.:

            discounts = (~(train_actor_out.cur_t == 0)).float() * self.im_discounting        
            behavior_logits_ls = [train_actor_out.im_policy_logits, train_actor_out.reset_policy_logits]
            target_logits_ls = [new_actor_out.im_policy_logits, new_actor_out.reset_policy_logits]
            actions_ls = [train_actor_out.im_action, train_actor_out.reset_action] 
            masks_ls = [im_mask, im_mask]  
            if not self.flags.im_cost_anneal: 
                c_ls = [self.flags.im_cost, self.flags.im_cost if not self.flags.reset_no_im_cost else 0.]
            else:
                c_ls = [self.flags.im_cost * self.anneal_c, (self.flags.im_cost if not self.flags.reset_no_im_cost else 0.) * self.anneal_c]
                        
            vtrace_returns = from_logits(
                behavior_logits_ls, target_logits_ls, actions_ls, masks_ls,
                discounts=discounts,
                rewards=rewards[:, :, 1],
                values=new_actor_out.baseline[:, :, 1],
                values_enc=new_actor_out.baseline_enc[:, :, 1] if new_actor_out.baseline_enc is not None else None,
                rv_tran=self.actor_net.rv_tran,
                enc_type=self.flags.critic_enc_type,                
                bootstrap_value=bootstrap_value[:, 1],
                flags=self.flags,
                lamb=self.flags.lamb,
                norm_stat=self.im_norm_stat
            )    
            if self.flags.critic_enc_type == 4: 
                self.b_im_norm_stat = update_b_norm_stat(
                    b_norm_stat=self.b_im_norm_stat,
                    target_baseline=vtrace_returns.vs,      
                )
            self.im_norm_stat = vtrace_returns.norm_stat

            advs = vtrace_returns.pg_advantages
            im_pg_loss = compute_policy_gradient_loss(target_logits_ls, actions_ls, masks_ls, c_ls, advs)   
            im_baseline_loss = self.flags.baseline_cost * compute_baseline_loss(
                new_actor_out=new_actor_out,
                ind=1,            
                target_baseline=vtrace_returns.vs,
                actor_net=self.actor_net,
                c=self.flags.im_cost if not self.flags.im_cost_anneal else self.flags.im_cost * self.anneal_c,
                enc_type=self.flags.critic_enc_type,
            )       
        else:
            im_pg_loss = torch.zeros(1, device=self.device)
            im_baseline_loss = torch.zeros(1, device=self.device)
            
        target_logits_ls = [new_actor_out.policy_logits, new_actor_out.im_policy_logits, new_actor_out.reset_policy_logits]
        masks_ls = [real_mask, im_mask, im_mask]    
        im_ent_c = self.flags.im_entropy_cost * (self.flags.real_im_cost + ((
            self.flags.im_cost if not self.flags.im_cost_anneal else self.flags.im_cost * self.anneal_c) if self.flags.im_cost > 0. else 0))
        c_ls = [self.flags.entropy_cost * self.flags.real_cost, im_ent_c, im_ent_c]
        entropy_loss = compute_entropy_loss(target_logits_ls, masks_ls, c_ls, self.flags.entropy_type==1)    

        reg_loss = self.flags.reg_cost * torch.sum(new_actor_out.reg_loss)
        total_loss = pg_loss + baseline_loss + entropy_loss + reg_loss         
            
        if self.flags.im_cost > 0.:
            total_loss = total_loss + im_pg_loss + im_baseline_loss   

        if self.flags.cur_cost > 0.:
            total_loss = total_loss + cur_pg_loss + cur_baseline_loss   

        losses = {"pg_loss": pg_loss,
                  "im_pg_loss": im_pg_loss,
                  "cur_pg_loss": cur_pg_loss,
                  "baseline_loss": baseline_loss,
                  "im_baseline_loss": im_baseline_loss,                  
                  "cur_baseline_loss": cur_baseline_loss,
                  "entropy_loss": entropy_loss,
                  "reg_loss": reg_loss,
                  "total_loss": total_loss
                  }
        return losses, train_actor_out

    def compute_stat(self, train_actor_out: TrainActorOut, losses: dict, total_norm: float, actor_id: torch.Tensor):
        """Update step, real_step and tot_eps; return training stat for printing"""
        if torch.any(train_actor_out.real_done):
            episode_returns = train_actor_out.episode_return[train_actor_out.real_done][:, 0]  
            episode_returns = tuple(episode_returns.detach().cpu().numpy())
            episode_lens = train_actor_out.episode_step[train_actor_out.real_done]  
            episode_lens = tuple(episode_lens.detach().cpu().numpy())            
            done_ids = actor_id.broadcast_to(train_actor_out.real_done.shape)[train_actor_out.real_done]
            done_ids = tuple(done_ids.detach().cpu().numpy())
        else:
            episode_returns, episode_lens, done_ids = (), (), ()
        self.last_returns.extend(episode_returns)
        rmean_episode_return = np.average(self.last_returns) if len(self.last_returns) > 0 else 0.

        if self.flags.im_cost > 0.:
            im_episode_returns = train_actor_out.episode_return[train_actor_out.cur_t==0][:, 1]  
            im_episode_returns = tuple(im_episode_returns.detach().cpu().numpy())
            self.last_im_returns.extend(im_episode_returns)
        rmean_im_episode_return = np.average(self.last_im_returns) if len(self.last_im_returns) > 0 else 0.

        if self.flags.cur_cost > 0.:
            cur_episode_returns = train_actor_out.episode_return[train_actor_out.real_done][:, 2]  
            cur_episode_returns = tuple(cur_episode_returns.detach().cpu().numpy())
            self.last_cur_returns.extend(cur_episode_returns)
        rmean_cur_episode_return = np.average(self.last_cur_returns) if len(self.last_cur_returns) > 0 else 0.

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
                    "rmean_cur_episode_return": rmean_cur_episode_return,
                    "episode_returns": episode_returns,  
                    "episode_lens": episode_lens,
                    "done_ids": done_ids,
                    "cur_real_step": cur_real_step,
                    "mean_plan_step": mean_plan_step,
                    "max_rollout_depth": max_rollout_depth,
                    "total_norm": total_norm
                    }

        if losses is not None: 
            for k, v in losses.items(): stats[k] = v.item()  

        if self.flags.return_norm_type != -1:
            stats["norm_diff"] = (self.norm_stat[1] - self.norm_stat[0]).item()
            stats["im_norm_diff"] = 0.
            stats["cur_norm_diff"] = 0.
            if self.im_norm_stat is not None:
                stats["im_norm_diff"] = (self.im_norm_stat[1] - self.im_norm_stat[0]).item()
            if self.cur_norm_stat is not None:
                stats["cur_norm_diff"] = (self.cur_norm_stat[1] - self.cur_norm_stat[0]).item()

        if self.flags.critic_enc_type == 4:
            stats["b_norm_diff"] = (self.b_norm_stat[1] - self.b_norm_stat[0]).item()
            stats["b_im_norm_diff"] = 0.
            stats["b_cur_norm_diff"] = 0.
            if self.b_im_norm_stat is not None:
                stats["b_im_norm_diff"] = (self.b_im_norm_stat[1] - self.b_im_norm_stat[0]).item()
            if self.b_cur_norm_stat is not None:
                stats["b_cur_norm_diff"] = (self.b_cur_norm_stat[1] - self.b_cur_norm_stat[0]).item()  

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
            self.check_point_path+".tmp",
        )
        os.replace(self.check_point_path+".tmp", self.check_point_path)
    
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
