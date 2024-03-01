import time
import timeit
import os
import numpy as np
import collections
import random
import copy
import traceback
import ray
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from thinker.core.vtrace import compute_v_trace
from thinker.core.file_writer import FileWriter
from thinker.core.module import guassian_kl_div
from thinker.actor_net import ActorNet
import thinker.util as util
from thinker.buffer import RetBuffer

def compute_baseline_loss(
    baseline,
    target_baseline,
    mask=None,
):
    target_baseline = target_baseline.detach()
    loss = (target_baseline - baseline)**2
    if mask is not None:
        loss = loss * mask
    return torch.sum(loss)

def compute_baseline_enc_loss(
    baseline_enc,
    target_baseline,
    rv_tran,
    enc_type,
    mask=None,
):
    target_baseline = target_baseline.detach()
    if enc_type == 1:
        baseline_enc = baseline_enc
        target_baseline_enc = rv_tran.encode(target_baseline)
        loss = (target_baseline_enc.detach() - baseline_enc)**2
    elif enc_type in [2, 3]:
        target_baseline_enc = rv_tran.encode(target_baseline)
        loss = (
            torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(baseline_enc, 0, 1),
                target=torch.flatten(target_baseline_enc, 0, 1).detach(),
            )            
        )
        loss = loss.view(baseline_enc.shape[:2])
    if mask is not None: loss = loss * mask
    return torch.sum(loss)

class SActorLearner:
    def __init__(self, ray_obj, actor_param, flags, actor_net=None, device=None):
        self.flags = flags
        self.time = flags.profile
        self._logger = util.logger()

        if flags.parallel_actor:
            self.actor_buffer = ray_obj["actor_buffer"]
            self.actor_param_buffer = ray_obj["actor_param_buffer"]
            self.actor_net = ActorNet(**actor_param)
            self.refresh_actor()
            self.actor_net.train(True)                
            if self.flags.gpu_learn_actor > 0. and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:           
                self.device = torch.device("cpu")
        else:
            assert actor_net is not None, "actor_net is required for non-parallel mode"
            assert device is not None, "device is required for non-parallel mode"
            self.actor_net = actor_net
            self.device = device

        if self.device == torch.device("cuda"):
            self._logger.info("Init. actor-learning: Using CUDA.")
        else:
            self._logger.info("Init. actor-learning: Not using CUDA.")

       # initialize learning setting

        if not self.flags.actor_use_rms:
            self.optimizer = torch.optim.Adam(
                self.actor_net.parameters(), lr=flags.actor_learning_rate, eps=flags.actor_adam_eps
            )
        else:
            self.optimizer = torch.optim.RMSprop(
                self.actor_net.parameters(),
                lr=flags.actor_learning_rate,
                momentum=0,
                eps=0.01,
                alpha=0.99,
            )

        self.step = 0
        self.tot_eps = 0
        self.real_step = 0

        lr_lambda = (
            lambda epoch: 1
            - min(epoch, self.flags.total_steps) / self.flags.total_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)        

        # other init. variables for consume_data
        max_actor_id = (
            self.flags.self_play_n * self.flags.env_n
        )
        self.ret_buffers = [RetBuffer(max_actor_id, mean_n=400)]
        if self.flags.im_cost > 0.:
            self.ret_buffers.append(RetBuffer(max_actor_id, mean_n=20000))
        if self.flags.cur_cost > 0.:
            self.ret_buffers.append(RetBuffer(max_actor_id, mean_n=400))      
        self.im_discounting = self.flags.discounting ** (1 / self.flags.rec_t)

        self.rewards_ls = ["re"]
        if flags.im_cost > 0.0:
            self.rewards_ls += ["im"]
        if flags.cur_cost > 0.0:
            self.rewards_ls += ["cur"]
        self.num_rewards = len(self.rewards_ls)

        self.norm_stats = [None,] * self.num_rewards
        self.anneal_c = 1
        self.n = 0

        self.crnorm = None

        self.ckp_path = os.path.join(flags.ckpdir, "ckp_actor.tar")
        if flags.ckp: self.load_checkpoint(self.ckp_path)

        # initialize file logs
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )
        
        # move network and optimizer to process device
        self.actor_net.to(self.device)
        util.optimizer_to(self.optimizer, self.device)        

        # variables for timing
        self.queue_n = 0
        self.timer = timeit.default_timer
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps = 0
        self.sps_buffer_n = 0
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.ckp_start_time = int(time.strftime("%M")) // 10
        self.disable_thinker = flags.wrapper_type == 1
    
        if self.flags.float16:
            self.scaler = GradScaler(init_scale=2**8)
        
        self.impact_enable = self.flags.impact_k > 1
        self.ppo = self.flags.ppo
        if self.ppo: 
            assert self.flags.impact_k > 1, f"For PPO mode, impact_k needs to be larger than 1, not {self.flags.impact_k}"
        if self.impact_enable:
            self.impact_n = self.flags.impact_n
            self.impact_k = self.flags.impact_k
            self.impact_b = self.flags.actor_batch_size
            if not self.flags.impact_syn:                
                assert (self.impact_n > self.impact_k and self.impact_n % self.impact_k == 0) or (
                    self.impact_n < self.impact_k and self.impact_k % self.impact_n == 0) or (
                    self.impact_n == self.impact_k
                    ), "impact_k and impact_n should be divisible"
                self.impact_update_freq = 1 if self.impact_k >= self.impact_n else self.impact_n // self.impact_k
                self.impact_update_time = 1 if self.impact_n >= self.impact_k else self.impact_k // self.impact_n                        
            else:
                self.impact_update_freq = self.impact_n
                self.impact_update_time = self.impact_k
            self.impact_update_tar_freq = self.flags.impact_update_tar_freq
            self.impact_t = 0
            self.impact_update_t = 0
            self.impact_buffer = None
            self.impact_buffer_n = self.impact_n * self.impact_b            

            self.tar_actor_net = ActorNet(**actor_param)
            self.tar_actor_net.to(self.device)
            self.tar_actor_net.train(False)
            self.update_target()
            self.kl_losses = collections.deque(maxlen=100)
            self.impact_is_abs = collections.deque(maxlen=100)
        self.dbg_adv = collections.deque(maxlen=100)
        self.dbg_start_time = self.timer()

    def learn_data(self):
        timing = util.Timings() if self.time else None
        data_ptr = self.actor_buffer.read.remote()                    
        try:
            while self.real_step < self.flags.total_steps:
                if timing is not None:
                    timing.reset()
                # get data remotely
           
                while True:
                    data = ray.get(data_ptr)
                    ray.internal.free(data_ptr)
                    data_ptr = self.actor_buffer.read.remote()                    
                    if data is not None:
                        break
                    time.sleep(0.001)
                    self.queue_n += 0.001
                if timing is not None:
                    timing.time("get_data")
         
                train_actor_out, initial_actor_state = data
                train_actor_out = util.tuple_map(
                    train_actor_out, lambda x: torch.tensor(x, device=self.device)
                )
                initial_actor_state = util.tuple_map(
                    initial_actor_state, lambda x: torch.tensor(x, device=self.device)
                )
                if timing is not None:
                    timing.time("convert_data")
                data = (train_actor_out, initial_actor_state)
                # start consume data
                self.consume_data(data, timing=timing)
                del train_actor_out, initial_actor_state, data
                
                self.actor_param_buffer.set_data.remote(
                    "actor_net", self.actor_net.get_weights()
                )
                if timing is not None:
                    timing.time("set weight")            
          
            self._logger.info("Terminating actor-learning thread")
            self.close()
            return True
        except Exception as e:
            self._logger.error(f"Exception detected in learn_actor: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            self.close()
            return True
        
    def update_target(self):
        for tar_module, new_module in zip(self.tar_actor_net.modules(), self.actor_net.modules()):
            if isinstance(tar_module, torch.nn.modules.batchnorm._BatchNorm):
                # Copy BatchNorm running mean and variance
                tar_module.running_mean = new_module.running_mean.clone()
                tar_module.running_var = new_module.running_var.clone()
            # Apply EMA to other parameters
            for tar_param, new_param in zip(tar_module.parameters(), new_module.parameters()):
                tar_param.data.mul_(self.flags.impact_update_lambda).add_(new_param.data, alpha=1 - self.flags.impact_update_lambda)

    def consume_data(self, data, timing=None):
        if not self.impact_enable: return self.consume_data_single(data, timing)
        train_actor_out, initial_actor_state = data
        TrainActorOut= type(train_actor_out)

        if self.impact_buffer is None:            
            out = {}
            for k in TrainActorOut._fields:
                out[k] = None
                v = getattr(train_actor_out, k)
                if v is None: continue
                out[k] = torch.zeros(size=(v.shape[0], self.impact_buffer_n) + v.shape[2:], dtype=v.dtype, device=self.device)
            self.impact_buffer = TrainActorOut(**out)            
            self.impact_buffer_actor_state = []
            for v in initial_actor_state:
                self.impact_buffer_actor_state.append(torch.zeros(size=(self.impact_buffer_n,)+v.shape[1:]), dtype=v.dtype, device=self.device)
            self.buffer_idx = 0
            self.buffer_wrote_n = 0

        for k in TrainActorOut._fields:
            v_ = getattr(self.impact_buffer, k)
            if v_ is None: continue           
            v = getattr(train_actor_out, k)
            v_[:, self.buffer_idx:self.buffer_idx+self.impact_b] = v
        for n, v in enumerate(initial_actor_state):
            self.impact_buffer_actor_state[n][self.buffer_idx:self.buffer_idx+self.impact_b] = v

        self.buffer_wrote_n = min(self.buffer_wrote_n + self.impact_b, self.impact_buffer_n) 
        self.buffer_idx = (self.buffer_idx + self.impact_b) % self.impact_buffer_n
        
        self.impact_t += 1        
        r = False                      
        if self.impact_t % self.impact_update_freq == 0:
            self.impact_early_stop = False
            for m in range(self.impact_update_time):
                if self.flags.impact_bmix:
                    ns = random.sample(range(self.buffer_wrote_n), self.buffer_wrote_n)
                    ns = [ns[i:i + self.impact_b] for i in range(0, len(ns), self.impact_b)]                       
                else:                    
                    ns = random.sample(range(self.buffer_wrote_n // self.impact_b), self.buffer_wrote_n // self.impact_b)
                    ns = [range(i*self.impact_b, i*self.impact_b+self.impact_b) for i in ns]         
                    
                for k, n in enumerate(ns):
                    out = {}
                    for k_ in TrainActorOut._fields:
                        out[k_] = None
                        v = getattr(self.impact_buffer, k_)
                        if v is None: continue           
                        out[k_] = v[:, n]
                    train_actor_out = TrainActorOut(**out)  

                    initial_actor_state = []       
                    for v in self.impact_buffer_actor_state:
                        initial_actor_state.append(v[n])
                    
                    data = (train_actor_out, initial_actor_state)
                    r = self.consume_data_single(data, timing=timing, first_iter=k<=self.impact_update_freq and m == 0, last_iter=k==len(ns)-1)
                    if self.impact_early_stop: break                
                self.impact_update_t += 1
                if self.impact_update_t % self.impact_update_tar_freq == 0: self.update_target()                  
                if self.impact_early_stop: break            
        return r

    def consume_data_single(self, data, timing=None, first_iter=True, last_iter=False):

        train_actor_out, initial_actor_state = data
        actor_id = train_actor_out.id
        T, B = train_actor_out.done.shape

        # compute losses
        out = self.compute_losses(
            train_actor_out, initial_actor_state, first_iter, last_iter
        )
        losses, train_actor_out = out
        total_loss = losses["total_loss"]
        if timing is not None:
            timing.time("compute loss")

        # gradient descent on loss
        self.optimizer.zero_grad()
        if self.flags.float16:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        if timing is not None:
            timing.time("compute gradient")

        optimize_params = self.optimizer.param_groups[0]["params"]
        if self.flags.float16:
            self.scaler.unscale_(self.optimizer)
        if self.flags.actor_grad_norm_clipping > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                optimize_params, self.flags.actor_grad_norm_clipping * T * B
            )
            total_norm = total_norm.detach().cpu().item()
        else:
            total_norm = util.compute_grad_norm(optimize_params)
        if timing is not None:
            timing.time("compute norm")

        if self.flags.float16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if timing is not None:
            timing.time("grad descent")

        self.scheduler.last_epoch = (
            max(self.real_step - 1, 0)
        )  # scheduler does not support setting epoch directly
        self.scheduler.step()
        self.anneal_c = max(1 - self.real_step / self.flags.total_steps, 0)
        
        if not self.impact_enable or first_iter:
            # statistic output
            for k in losses: losses[k] = losses[k] / T / B
            total_norm = total_norm / T / B
            stats = self.compute_stat(train_actor_out, losses, total_norm, actor_id)
            stats["sps"] = self.sps

            # write to log file
            self.plogger.log(stats)

            # print statistics
            if self.timer() - self.start_time > 5:
                self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())
                self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
                self.sps = (
                    self.sps_buffer[self.sps_buffer_n - 1][0]
                    - self.sps_buffer[self.sps_buffer_n][0]
                ) / (
                    self.sps_buffer[self.sps_buffer_n - 1][1]
                    - self.sps_buffer[self.sps_buffer_n][1]
                )
                tot_sps = (self.step - self.sps_start_step) / (
                    self.timer() - self.sps_start_time
                )
                print_str = (
                    "[%s] Steps %i @ %.1f SPS (%.1f). (T_q: %.2f) Eps %i. Ret %f (%f/%f). Loss %.2f"
                    % (
                        self.flags.xpid,
                        self.real_step,
                        self.sps,
                        tot_sps,
                        self.queue_n,
                        self.tot_eps,
                        stats["rmean_episode_return"],
                        stats.get("rmean_im_episode_return", 0.),
                        stats.get("rmean_cur_episode_return", 0.),
                        total_loss/T/B,
                    )
                )
                print_stats = [
                    "actor/pg_loss",
                    "actor/entropy_loss",
                    "actor/reg_loss",
                    "actor/total_norm",
                    "actor/mean_abs_v",
                ]
                for k in print_stats:
                    print_str += " %s %.2f" % (k.replace("actor/", ""), stats[k])
                if self.flags.return_norm_type in [0, 1]:
                    print_str += " norm_diff %.4f/%.4f" % (
                        stats["actor/norm_diff"],
                        stats.get("actor/im_norm_diff", 0.),
                    )
                if self.flags.cur_return_norm_type in [0, 1]:
                    print_str += " cur_norm_diff %.4f" % (
                        stats.get("actor/cur_norm_diff", 0.),
                    )
                if self.impact_enable:
                    print_str += " kl_beta %.4f" % self.actor_net.kl_beta
                    print_str += " kl_loss %.4f" % losses["kl_loss"]
                    print_str += " is_abs %.4f" % np.mean(self.impact_is_abs)

                # dbg_adv = torch.concat(list(self.dbg_adv))
                # print_str += " dbg_adv mean %.4f std %.4f abs %.4f" % (torch.mean(dbg_adv), torch.std(dbg_adv), torch.mean(torch.abs(dbg_adv)))

                self._logger.info(print_str)
                self.start_time = self.timer()
                self.queue_n = 0
                if timing is not None:
                    print(timing.summary())

            if int(time.strftime("%M")) // 10 != self.ckp_start_time:
                self.save_checkpoint()
                self.ckp_start_time = int(time.strftime("%M")) // 10
            del train_actor_out, losses, total_loss, stats, total_norm
        else:
            del train_actor_out, losses, total_loss, total_norm

        if timing is not None:
            timing.time("misc")
        
        torch.cuda.empty_cache()

        # update shared buffer's weights
        self.n += 1
        r = self.real_step > self.flags.total_steps
        return r

    def compute_losses(self, train_actor_out, initial_actor_state, first_iter=True, last_iter=False):
        # compute loss and then discard the first step in train_actor_out

        T, B = train_actor_out.done.shape
        T = T - 1        
        
        if self.disable_thinker:
            clamp_action = train_actor_out.pri[1:]
        else:
            clamp_action = (train_actor_out.pri[1:], train_actor_out.reset[1:])
        
        new_actor_out, _ = self.actor_net(
            train_actor_out, 
            initial_actor_state,
            clamp_action = clamp_action,
            compute_loss = True,
        )
        if self.impact_enable:
            with torch.no_grad():
                tar_actor_out, _ = self.tar_actor_net(
                    train_actor_out, 
                    initial_actor_state,
                    clamp_action = clamp_action,
                    compute_loss = False,
                )

        # Take final value function slice for bootstrapping.
        if self.impact_enable:
            bootstrap_value = tar_actor_out.baseline[-1]
        else:
            bootstrap_value = new_actor_out.baseline[-1]        
    
        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        train_actor_out = util.tuple_map(train_actor_out, lambda x: x[1:])
        new_actor_out = util.tuple_map(new_actor_out, lambda x: x[:-1])

        if self.impact_enable:
            # record base policy for impact / ppo
            tar_actor_out = util.tuple_map(tar_actor_out, lambda x: x[:-1])
            base_actor_out = tar_actor_out
            if self.actor_net.discrete_action:
                base_pri_logits = base_actor_out.pri_param.detach()
            else:
                pri_param = base_actor_out.pri_param.detach()
                base_pri_mean = pri_param[:, :, :, 0]
                base_pri_log_var = pri_param[:, :, :, 1]
            if not self.disable_thinker:
                base_reset_logits = base_actor_out.reset_logits.detach()
        rewards = train_actor_out.reward

        # compute advantage and baseline        
        pg_losses = []
        baseline_losses = []
        discounts = [(~train_actor_out.done).float() * self.im_discounting]
        masks = [None]

        last_step_real = (train_actor_out.step_status == 0) | (train_actor_out.step_status == 3)
        next_step_real = (train_actor_out.step_status == 2) | (train_actor_out.step_status == 3)
        
        if self.flags.im_cost > 0.:
            discounts.append((~next_step_real).float() * self.im_discounting)            
            masks.append((~last_step_real).float())
        if self.flags.cur_cost > 0.:
            discounts.append((~train_actor_out.done).float() * self.im_discounting)            
            masks.append(None)

        if not self.impact_enable or self.flags.impact_v_trace:
            log_rhos = new_actor_out.c_action_log_prob - train_actor_out.c_action_log_prob
        elif not self.ppo:
            log_rhos = tar_actor_out.c_action_log_prob - train_actor_out.c_action_log_prob
        elif self.ppo:
            log_rhos = torch.zeros_like(train_actor_out.c_action_log_prob)

        for i in range(self.num_rewards):
            prefix = self.rewards_ls[i]
            prefix_rewards = rewards[:, :, i]
            
            if self.flags.entropy_r_cost > 0. and prefix == "re":
                prefix_rewards[last_step_real] += -self.flags.entropy_r_cost * train_actor_out.c_action_log_prob[last_step_real]

            if prefix == "cur":
                return_norm_type=self.flags.cur_return_norm_type 
                cur_gate = train_actor_out.cur_gate
                prefix_rewards, self.crnorm = util.cur_reward_norm(prefix_rewards, self.crnorm, cur_gate, self.flags)      
            else:    
                return_norm_type=self.flags.return_norm_type 

            if not self.impact_enable:
                values = new_actor_out.baseline[:, :, i]
            else:
                values = tar_actor_out.baseline[:, :, i]
            v_trace = compute_v_trace(
                log_rhos=log_rhos,
                discounts=discounts[i],
                rewards=prefix_rewards,
                values=values,
                bootstrap_value=bootstrap_value[:, i],
                return_norm_type=return_norm_type,
                norm_stat=self.norm_stats[i], 
                lamb=self.flags.v_trace_lamb,
            )                
            self.norm_stats[i] = v_trace.norm_stat
            if self.impact_enable:                
                if not self.ppo:
                    beta = torch.log(torch.tensor(self.flags.impact_beta, device=self.device))
                    log_is_de = torch.maximum(tar_actor_out.c_action_log_prob, train_actor_out.c_action_log_prob + beta)
                else:
                    log_is_de = train_actor_out.c_action_log_prob
                adv = v_trace.pg_advantages_nois.detach()
                log_is_de = log_is_de.detach()
                vs = v_trace.vs.detach()

            if not self.impact_enable:
                adv = v_trace.pg_advantages.detach()
                pg_loss = -adv * new_actor_out.c_action_log_prob
            else:                
                log_is = new_actor_out.c_action_log_prob - log_is_de
                unclipped_is = torch.exp(log_is) 
                self.impact_is_abs.append(torch.mean(torch.abs(unclipped_is-1)).detach().item())
                clipped_is = torch.clamp(unclipped_is, 1-self.flags.impact_clip, 1+self.flags.impact_clip)
                pg_loss = -torch.minimum(unclipped_is * adv, clipped_is * adv)

            if masks[i] is not None: pg_loss = pg_loss * masks[i]
            pg_loss = torch.sum(pg_loss)

            vs = v_trace.vs if not self.impact_enable else vs
            pg_losses.append(pg_loss)
            if self.flags.critic_enc_type == 0:
                baseline_loss = compute_baseline_loss(
                    baseline=new_actor_out.baseline[:, :, i],
                    target_baseline=vs,
                    mask=masks[i]
                )
            else:
                baseline_loss = compute_baseline_enc_loss(
                    baseline_enc=new_actor_out.baseline_enc[:, :, i],
                    target_baseline=vs,
                    rv_tran=self.actor_net.rv_tran,
                    enc_type=self.flags.critic_enc_type,
                    mask=masks[i]
                )

            baseline_losses.append(baseline_loss)

        # sum all the losses
        total_loss = pg_losses[0] / self.actor_net.dim_actions
        total_loss += self.flags.baseline_cost * baseline_losses[0]

        losses = {
            "pg_loss": pg_losses[0],
            "baseline_loss": baseline_losses[0]
        }
        n = 0
        for prefix in ["im", "cur"]:
            cost = getattr(self.flags, "%s_cost" % prefix)
            if cost > 0.:
                n += 1
                if getattr(self.flags, "%s_cost_anneal" % prefix):
                    cost *= self.anneal_c
                total_loss += cost * pg_losses[n] / self.actor_net.dim_actions
                total_loss += (cost * self.flags.baseline_cost * 
                            baseline_losses[n])
                losses["%s_pg_loss" % prefix] = pg_losses[n]
                losses["%s_baseline_loss" % prefix] = baseline_losses[n]

        # process entropy loss

        f_entropy_loss = new_actor_out.entropy_loss
        entropy_loss = f_entropy_loss * last_step_real.float()
        entropy_loss = torch.sum(entropy_loss)        
        losses["entropy_loss"] = entropy_loss
        total_loss += self.flags.entropy_cost * entropy_loss / self.actor_net.dim_actions

        if not self.disable_thinker:
            im_entropy_loss = f_entropy_loss * (~last_step_real).float()
            im_entropy_loss = torch.sum(im_entropy_loss)
            total_loss += self.flags.im_entropy_cost * im_entropy_loss
            losses["im_entropy_loss"] = im_entropy_loss / self.actor_net.dim_actions

        reg_loss = torch.sum(new_actor_out.reg_loss)
        losses["reg_loss"] = reg_loss
        total_loss += self.flags.reg_cost * reg_loss

        if self.impact_enable:
            if self.actor_net.discrete_action:
                tar_pri_log_prob = F.log_softmax(base_pri_logits, dim=-1)
                pri_log_prob = F.log_softmax(new_actor_out.pri_param, dim=-1)
                pri_kl_loss = F.kl_div(pri_log_prob, tar_pri_log_prob, reduction="none", log_target=True)
                pri_kl_loss = torch.sum(pri_kl_loss, dim=-1)
            else:
                pri_kl_loss = guassian_kl_div(
                    base_pri_mean, 
                    base_pri_log_var,
                    new_actor_out.pri_param[:, :, :, 0],
                    new_actor_out.pri_param[:, :, :, 1]
                )            
            pri_kl_loss = torch.sum(pri_kl_loss)
            # if impact_first_sample: 
                # print("new", new_actor_out.pri_param[:2, 4, :3])
                # print("old", train_actor_out.pri_param[:2, 4, :3])
                # print("pri_kl_loss", pri_kl_loss)
            kl_loss = pri_kl_loss

            if not self.disable_thinker:                
                tar_reset_log_prob = F.log_softmax(base_reset_logits, dim=-1)
                reset_log_prob = F.log_softmax(new_actor_out.reset_logits, dim=-1)
                reset_kl_loss = F.kl_div(reset_log_prob, tar_reset_log_prob, reduction="sum", log_target=True)
                kl_loss += reset_kl_loss

            if self.flags.impact_kl_coef > 0.:
                total_loss += self.flags.impact_kl_coef * self.actor_net.kl_beta * kl_loss         
                avg_kl_loss = kl_loss / T / B  
                if last_iter:                
                    if avg_kl_loss < self.flags.impact_kl_targ / 1.5:
                        self.actor_net.kl_beta /= 2
                    elif avg_kl_loss > self.flags.impact_kl_targ * 1.5:
                        self.actor_net.kl_beta *= 2
                if self.flags.impact_early_stop:
                    if avg_kl_loss > self.flags.impact_kl_targ:
                        self.impact_early_stop = True
                self.actor_net.kl_beta = torch.clamp(self.actor_net.kl_beta, 1e-6, 1e3)
            self.kl_losses.append(kl_loss.item())            
            losses["kl_loss"] = np.mean(self.kl_losses)
        losses["total_loss"] = total_loss

        return losses, train_actor_out

    def compute_stat(self, train_actor_out, losses, total_norm, actor_id):
        """Update step, real_step and tot_eps; return training stat for printing"""
        stats = {}
        T, B, *_ = train_actor_out.episode_return.shape
        last_step_real = (train_actor_out.step_status == 0) | (train_actor_out.step_status == 3)
        next_step_real = (train_actor_out.step_status == 2) | (train_actor_out.step_status == 3)

        # extract episode_returns
        if torch.any(train_actor_out.real_done):            
            episode_returns = train_actor_out.episode_return[train_actor_out.real_done][
                :, 0
            ]
            episode_returns = tuple(episode_returns.detach().cpu().numpy())
            episode_lens = train_actor_out.episode_step[train_actor_out.real_done]
            episode_lens = tuple(episode_lens.detach().cpu().numpy())
            done_ids = actor_id.broadcast_to(train_actor_out.real_done.shape)[
                train_actor_out.real_done
            ]
            done_ids = tuple(done_ids.detach().cpu().numpy())
        else:
            episode_returns, episode_lens, done_ids = (), (), ()

        self.ret_buffers[0].insert(episode_returns, done_ids)
        stats = {"rmean_episode_return": self.ret_buffers[0].get_mean()}

        for prefix in ["im", "cur"]:            
            if prefix == "im":
                done = next_step_real
            elif prefix == "cur":
                done = train_actor_out.real_done
            
            if prefix in self.rewards_ls:            
                n = self.rewards_ls.index(prefix)
                self.ret_buffers[n].insert_raw(
                    train_actor_out.episode_return,
                    ind=n,
                    actor_id=actor_id,
                    done=done,
                )
                r = self.ret_buffers[n].get_mean()
                stats["rmean_%s_episode_return" % prefix] = r

        if not self.disable_thinker:
            max_rollout_depth = (
                (train_actor_out.max_rollout_depth[last_step_real & ~next_step_real])
                .detach()
                .cpu()
                .numpy()
            )
            max_rollout_depth = (
                np.average(max_rollout_depth) if len(max_rollout_depth) > 0 else 0.0
            )
            stats["max_rollout_depth"] = max_rollout_depth

        self.step += T * B
        self.real_step += torch.sum(last_step_real).item()
        self.tot_eps += torch.sum(train_actor_out.real_done).item()
        mean_abs_v = torch.mean(torch.abs(train_actor_out.baseline)).item()

        stats.update({
            "step": self.step,
            "real_step": self.real_step,
            "tot_eps": self.tot_eps,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
            "actor/total_norm": total_norm,
            "actor/mean_abs_v": mean_abs_v,
        })

        if losses is not None:
            for k, v in losses.items():
                if v is not None:
                    stats["actor/"+k] = v.item()

        if self.flags.return_norm_type in [0, 1]:
            n = self.rewards_ls.index("re")
            stats["actor/norm_diff"] = (
                self.norm_stats[n][1] - self.norm_stats[n][0]
                ).item()            
            stats["norm_rmean_episode_return"] = (stats["rmean_episode_return"] / self.norm_stats[n][2]).item()
            if "im" in self.rewards_ls:
                n = self.rewards_ls.index("im")
                stats["actor/im_norm_diff"] = (
                    self.norm_stats[n][1] - self.norm_stats[n][0]
                ).item()
                stats["norm_rmean_im_episode_return"] = (stats["rmean_im_episode_return"] / self.norm_stats[n][2]).item()
        if self.flags.cur_return_norm_type in [0, 1]:
            if "cur" in self.rewards_ls:
                n = self.rewards_ls.index("cur")
                stats["actor/cur_norm_diff"] = (
                    self.norm_stats[n][1] - self.norm_stats[n][0]
                ).item()
                stats["norm_rmean_cur_episode_return"] = (stats["rmean_cur_episode_return"] / self.norm_stats[n][2]).item()
        return stats

    def save_checkpoint(self):
        self._logger.info("Saving actor checkpoint to %s" % self.ckp_path)
        d = {
                "step": self.step,
                "real_step": self.real_step,
                "tot_eps": self.tot_eps,
                "ret_buffers": self.ret_buffers,
                "norm_stats": self.norm_stats,
                "crnorm": self.crnorm, 
                "actor_net_optimizer_state_dict": self.optimizer.state_dict(),
                "actor_net_scheduler_state_dict": self.scheduler.state_dict(),
                "actor_net_state_dict": self.actor_net.state_dict(),                
                "flags": vars(self.flags),
            }      
        try:
            torch.save(d, self.ckp_path + ".tmp")
            os.replace(self.ckp_path + ".tmp", self.ckp_path)
        except:       
            pass

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.ret_buffers = train_checkpoint["ret_buffers"]
        self.norm_stats = train_checkpoint["norm_stats"]
        self.crnorm = train_checkpoint["crnorm"]
        self.optimizer.load_state_dict(
            train_checkpoint["actor_net_optimizer_state_dict"]
        )
        self.scheduler.load_state_dict(
            train_checkpoint["actor_net_scheduler_state_dict"]
        )
        self.actor_net.set_weights(train_checkpoint["actor_net_state_dict"])
        self._logger.info("Loaded actor checkpoint from %s" % ckp_path)

    def refresh_actor(self):
        while True:
            weights = ray.get(
                self.actor_param_buffer.get_data.remote("actor_net")
            )  
            if weights is not None:
                self.actor_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  

    def close(self):
        self.actor_buffer.set_finish.remote()
        self.plogger.close()


@ray.remote
class ActorLearner(SActorLearner):
    pass
