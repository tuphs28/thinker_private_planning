import os
import numpy as np
import argparse
import time
import timeit
import traceback
import ray
import torch
import torch.nn.functional as F
from thinker.core.file_writer import FileWriter
from thinker.buffer import GeneralBuffer, ModelBuffer
from thinker.net import ModelNet
from thinker.self_play import SelfPlayWorker, TrainModelOut, PO_MODEL
from thinker.env import Environment
import thinker.util as util

def compute_cross_entropy_loss(logits, target_logits, is_weights, mask=None):
    k, b, *_ = logits.shape
    target_policy = F.softmax(target_logits, dim=-1)
    loss = torch.nn.CrossEntropyLoss(reduction="none")(
                    input = torch.flatten(logits, 0, 1),
                    target = torch.flatten(target_policy, 0, 1)
                )
    loss = loss.view(k, b)
    if mask is not None: loss = loss * mask
    loss = torch.sum(loss, dim=0)
    loss = is_weights * loss
    return torch.sum(loss)

@ray.remote
class ModelLearner():
    def __init__(self, param_buffer: GeneralBuffer, model_buffer: ModelBuffer, rank: int, flags: argparse.Namespace):
        self.param_buffer = param_buffer
        self.model_buffer = model_buffer
        self.rank = rank
        self.flags = flags
        self._logger = util.logger()
        self.wlogger = util.Wandb(flags, subname='_model') if flags.use_wandb else None

        env = Environment(flags)        
        self.model_net = ModelNet(obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags)
        env.close()
        self.model_net.train(True)        

        # initialize learning setting

        if not self.flags.disable_cuda and torch.cuda.is_available():
            self._logger.info("Model-learning: Using CUDA.")
            self.device = torch.device("cuda")
        else:
            self._logger.info("Model-learning: Not using CUDA.")
            self.device = torch.device("cpu")

        self.step = 0
        self.real_step = 0
        
        lr_lambda = lambda epoch: 1 - min(epoch, self.flags.total_steps) / self.flags.total_steps
        if self.flags.duel_net:
            self.optimizer_m = torch.optim.Adam(self.model_net.model_net.parameters(),lr=flags.model_learning_rate)            
            self.scheduler_m = torch.optim.lr_scheduler.LambdaLR(self.optimizer_m, lr_lambda)
        self.optimizer_p = torch.optim.Adam(self.model_net.pred_net.parameters(),lr=flags.model_learning_rate)        
        self.scheduler_p = torch.optim.lr_scheduler.LambdaLR(self.optimizer_p, lr_lambda)

        if self.flags.preload_model and not flags.load_checkpoint:
            checkpoint = torch.load(self.flags.preload_model, map_location=torch.device('cpu'))
            self.model_net.set_weights(checkpoint["model_state_dict" if "model_state_dict" in checkpoint else "model_net_state_dict"])  
            self._logger.info("Loadded model network from %s" % self.flags.preload_model)
            if "model_net_optimizer_state_dict" in checkpoint:
                if self.flags.duel_net:
                    self.optimizer_m.load_state_dict(checkpoint["model_net_optimizer_m_state_dict"])  
                self.optimizer_p.load_state_dict(checkpoint["model_net_optimizer_p_state_dict"])  
                self._logger.info("Loadded model network's optimizer from %s" % self.flags.preload_model)            

        if flags.load_checkpoint:
            self.load_checkpoint(os.path.join(flags.load_checkpoint, "ckp_model.tar"))
            self.flags.savedir = os.path.split(self.flags.load_checkpoint)[0]
            self.flags.xpid = os.path.split(self.flags.load_checkpoint)[-1]    

        self.flags.git_revision = util.get_git_revision_hash()
        self.plogger = FileWriter(xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir, suffix="_model")

        self.check_point_path = "%s/%s/%s" % (flags.savedir, flags.xpid, "ckp_model.tar")

        # set shared buffer's weights
        self.param_buffer.set_data.remote("model_net", self.model_net.get_weights())

        # move network and optimizer to process device
        self.model_net.to(self.device)
        if self.flags.duel_net: util.optimizer_to(self.optimizer_m, self.device)
        util.optimizer_to(self.optimizer_p, self.device)

        # model tester
        self.test_buffer = GeneralBuffer.remote()   
        self.model_tester = [SelfPlayWorker.remote(
            param_buffer=param_buffer, 
            actor_buffer=None, 
            model_buffer=None, 
            test_buffer=self.test_buffer, 
            policy=self.flags.test_policy_type, 
            policy_params=None, 
            rank=n+1, 
            num_p_actors=1,
            flags=flags) for n in range(10)]

    def learn_data(self):
        try:
            timer = timeit.default_timer
            start_step = self.step
            start_step_test = self.step
            start_time = timer()
            ckp_start_time = int(time.strftime("%M")) // 10
            last_psteps = 0
            
            r_tester = None
            all_returns = None
            
            numel_per_step = self.flags.model_batch_size * (
                    self.flags.model_k_step_return if not self.flags.model_batch_mode else 
                    self.flags.model_unroll_length)

            max_diff = 200000
            # stop training the model at the last 200k real steps

            n = 0
            if self.flags.float16:
                self.scaler = torch.cuda.amp.GradScaler(init_scale=2**8)

            while (self.real_step < self.flags.total_steps - max_diff):                    
                c = min(self.real_step, self.flags.total_steps) / self.flags.total_steps
                beta = self.flags.priority_beta * (1 - c) + 1. * c

                # get data remotely    
                while (True):
                    data = ray.get(self.model_buffer.read.remote(beta))
                    if data is not None: break            
                    time.sleep(0.01)
                    if timer() - start_time > 5:
                        tran_n = ray.get(self.model_buffer.get_processed_n.remote())
                        self._logger.info("Preloading: %d/%d" % (tran_n, self.flags.model_warm_up_n))
                        start_time = timer()                

                # start consume data
                train_model_out, is_weights, inds, new_psteps = data            
                self.real_step += new_psteps - last_psteps            
                last_psteps = new_psteps

                # move the data to the process device
                train_model_out = util.tuple_map(train_model_out, lambda x:x.to(self.device))
                is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
                target = self.prepare_data(train_model_out)

                if self.flags.duel_net:                            
                    # compute losses for model_net
                    if self.flags.float16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):      
                            losses_m, pred_xs = self.compute_losses_m(train_model_out, target, is_weights)
                    else:
                        losses_m, pred_xs = self.compute_losses_m(train_model_out, target, is_weights)
                    total_norm_m = self.gradient_step(losses_m["total_loss_m"], self.optimizer_m, self.scheduler_m)                    
                else:
                    losses_m = {}
                    total_norm_m = torch.zeros(1, device=self.device)
                    pred_xs = None

                if self.flags.float16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):      
                        losses_p, priorities = self.compute_losses_p(train_model_out, target, is_weights, pred_xs)
                else:
                    losses_p, priorities = self.compute_losses_p(train_model_out, target, is_weights, pred_xs)
                total_norm_p = self.gradient_step(losses_p["total_loss_p"], self.optimizer_p, self.scheduler_p)
                if self.flags.priority_alpha > 0:
                    self.model_buffer.update_priority.remote(inds, priorities)                
                self.step += numel_per_step

                losses = losses_m
                losses.update(losses_p)
                # print statistics
                if timer() - start_time > 5:
                    sps = (self.step - start_step) / (timer() - start_time)                
                    print_str =  "Steps %i (%i:%i[%.1f]) @ %.1f SPS. Model return mean (std) %f (%f) norm_m %.2f norm_p %.2f" % (
                                    n, 
                                    self.real_step, 
                                    self.step, 
                                    self.step_per_transition(), 
                                    sps, 
                                    np.mean(all_returns) if all_returns is not None else 0.,
                                    np.std(all_returns) if all_returns is not None else 0.,
                                    total_norm_m.item(),
                                    total_norm_p.item())
                    print_stats = ["total_loss_m", "total_loss_p", "vs_loss", "logits_loss", 
                                   "rs_loss", "sup_loss", "img_loss", "done_loss"]
                    for k in print_stats: 
                        if k in losses and losses[k] is not None:
                            print_str += " %s %.6f" % (k, losses[k].item() / numel_per_step)
                    self._logger.info(print_str)
                    start_step = self.step
                    start_time = timer()      

                    # write to log file
                    stats = {"step": self.step,
                            "real_step": self.real_step,
                            "model_returns_mean": np.mean(all_returns) if all_returns is not None else None,
                            "model_returns_std": np.std(all_returns)/np.sqrt(len(all_returns)) if all_returns is not None else None,
                            "model_total_norm_m": total_norm_m.item(),
                            "model_total_norm_p": total_norm_p.item()}
                    for k in print_stats: 
                        stats[k] = losses[k].item() / numel_per_step if k in losses and losses[k] is not None else None

                    self.plogger.log(stats)
                    if self.flags.use_wandb:
                        self.wlogger.wandb.log(stats, step=stats['real_step'])
                
                if int(time.strftime("%M")) // 10 != ckp_start_time:
                    self.save_checkpoint()
                    ckp_start_time = int(time.strftime("%M")) // 10
                
                # update shared buffer's weights
                if n % 1 == 0:
                    self.param_buffer.set_data.remote("model_net", self.model_net.get_weights())

                # test the model policy returns
                if self.step - start_step_test > 250000 * self.flags.rec_t:
                    start_step_test = self.step
                    if r_tester is not None: 
                        ray.get(r_tester)[0]
                        all_returns = ray.get(self.test_buffer.get_data.remote("episode_returns"))
                        self.test_buffer.set_data.remote("episode_returns", [])
                        #self._logger.info("Steps %i Model policy returns for %i episodes: Mean (Std.) %.4f (%.4f)" % 
                        #    (n, len(all_returns), np.mean(all_returns), np.std(all_returns)/np.sqrt(len(all_returns))))
                    r_tester = [x.gen_data.remote(test_eps_n=20, verbose=False) for x in self.model_tester]                                        

                # control the number of learning step per transition
                while (self.flags.model_max_step_per_transition > 0 and 
                    self.step_per_transition() > self.flags.model_max_step_per_transition):
                    time.sleep(0.1)
                    self.param_buffer.update_dict_item.remote("self_play_signals", "halt", False)
                    new_psteps = ray.get(self.model_buffer.get_processed_n.remote())
                    self.real_step += new_psteps - last_psteps            
                    last_psteps = new_psteps                
                
                if self.flags.model_min_step_per_transition > 0:
                    if self.step_per_transition() < self.flags.model_min_step_per_transition:
                        self.param_buffer.update_dict_item.remote("self_play_signals", "halt", True)
                    else:
                        self.param_buffer.update_dict_item.remote("self_play_signals", "halt", False)

                n += 1            
            
            self.close(0)  
            return True        
        
        except Exception as e:
            self._logger.error("Exception detected in learn_model")
            self._logger.error(traceback.format_exc())
            self.close(1)            
            return False

    def compute_rs_loss(self, target, rs, r_enc_logits, reward_tran, is_weights):
        k, b = self.flags.model_k_step_return, target["rewards"].shape[1]
        done_mask = target["done_mask"]
        if not self.flags.reward_transform:
                rs_loss = (rs - target["rewards"]) ** 2
        else:
            _, target_rs_enc_v = reward_tran.encode(target["rewards"])
            rs_loss = torch.nn.CrossEntropyLoss(reduction="none")(
                    input = torch.flatten(r_enc_logits, 0, 1),
                    target = torch.flatten(target_rs_enc_v, 0, 1),
                    )
            rs_loss = rs_loss.view(k, b)
        rs_loss = rs_loss * done_mask[:-1]
        rs_loss = torch.sum(rs_loss, dim=0)
        rs_loss = rs_loss * is_weights
        rs_loss = torch.sum(rs_loss)
        return rs_loss
    
    def compute_done_loss(self, target, pred_done_logits, is_weights):
        if self.flags.model_done_loss_cost > 0.:
            done_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
                pred_done_logits, target["dones"])       
            done_loss = done_loss * (~target["trun_done"]).float()[:-1]
            done_loss = torch.sum(done_loss, dim=0)
            done_loss = done_loss * is_weights
            done_loss = torch.sum(done_loss)
        else:
            done_loss = None
        return done_loss

    def compute_losses_m(self, train_model_out, target, is_weights):
        k, b = self.flags.model_k_step_return, train_model_out.gym_env_out.shape[1]
        out = self.model_net.model_net.forward(
            x=train_model_out.gym_env_out[0].float()/255.,
            actions=train_model_out.action[:k+1],
            one_hot=False
        )
        rs_loss = self.compute_rs_loss(target, out.rs, out.r_enc_logits, 
                                       self.model_net.model_net.reward_tran,
                                       is_weights)        
        done_loss = self.compute_done_loss(target, out.done_logits, is_weights)
        img_loss = torch.mean(torch.square(target["xs"] - out.xs), dim=(2, 3, 4))
        img_loss = img_loss * target["done_mask"][1:]
        img_loss = torch.sum(img_loss, dim=0)
        img_loss = img_loss * is_weights
        img_loss = torch.sum(img_loss)

        total_loss = self.flags.model_rs_loss_cost * rs_loss
        total_loss = total_loss + self.flags.model_img_loss_cost * img_loss
        if self.flags.model_done_loss_cost > 0.:
            total_loss = total_loss + self.flags.model_done_loss_cost * done_loss
        return {"rs_loss": rs_loss,
                "done_loss": done_loss,
                "img_loss": img_loss,
                "total_loss_m": total_loss,
            }, out.xs.detach()
    
    def compute_losses_p(self, train_model_out, target, is_weights, pred_xs):
        k, b = self.flags.model_k_step_return, train_model_out.gym_env_out.shape[1]
        first_x = train_model_out.gym_env_out[[0]].float()/255.
        if self.flags.duel_net:            
            xs = torch.concat([first_x, pred_xs], dim=0)
        else:
            xs = torch.concat([first_x, target["xs"]], dim=0)
        
        if self.flags.perfect_model:
            out = self.model_net.pred_net.forward(
                xs=xs[:k].view((k*b,) + xs.shape[2:]),
                actions=train_model_out.action[:k].view(1, k*b),  
                one_hot=False)
            vs = out.vs.view(k, b)   
            v_enc_logits = util.safe_view(out.v_enc_logits, (k, b, -1))
            logits = out.logits.view(k, b, -1)
        else:
            out = self.model_net.pred_net.forward(
                xs=xs[:k+1], # s_0, ..., s_k   
                actions=train_model_out.action[:k+1], # a_-1, ..., a_k-1      
                one_hot=False)
            vs = out.vs[:-1].view(k, b)   
            if out.v_enc_logits is not None:
                v_enc_logits = util.safe_view(out.v_enc_logits[:-1], (k, b, -1))
            else:
                v_enc_logits = None
            logits = out.logits[:-1].view(k, b, -1)
        
        done_mask = target["done_mask"]
        if self.model_net.pred_net.predict_rd:
            rs_loss = self.compute_rs_loss(target, out.rs, out.r_enc_logits, 
                                           self.model_net.pred_net.reward_tran, 
                                           is_weights)        
            done_loss = self.compute_done_loss(target, out.done_logits, is_weights)

        # compute vs loss
        if not self.flags.reward_transform:
            vs_loss = (vs[:k] - target["vs"].detach()) ** 2
        else:
            _, target_vs_enc_v = self.model_net.pred_net.reward_tran.encode(target["vs"])
            vs_loss = torch.nn.CrossEntropyLoss(reduction="none")(
                        input = torch.flatten(v_enc_logits[:k], 0, 1),
                        target = torch.flatten(target_vs_enc_v.detach(), 0, 1)
                      )
            vs_loss = vs_loss.view(k, b)
        vs_loss = vs_loss * done_mask[:-1]
        vs_loss = torch.sum(vs_loss, dim=0)
        vs_loss = vs_loss * is_weights
        vs_loss = torch.sum(vs_loss)

        # compute logit loss
        logits_loss = compute_cross_entropy_loss(
            logits, target["logits"].detach(), is_weights, mask=done_mask[:-1]) 
        
        # compute sup loss
        if self.flags.model_sup_loss_cost > 0.:
            tgt = out.true_zs[1:].detach().flatten(2)
            src = out.pred_zs[1:].flatten(2)
            if self.flags.model_supervise_type == 1:                
                sup_loss = -torch.nn.CosineSimilarity(dim=2, eps=1e-6)(tgt, src)
            elif self.flags.model_supervise_type == 2:
                sup_loss = torch.mean(torch.square(tgt-src), dim=-1)
            else:
                raise Exception("supervise loss type not supported:", self.flags.model_supervise_type)
            sup_loss = sup_loss * done_mask[1:]
            sup_loss = torch.sum(sup_loss, dim=0)
            sup_loss = sup_loss * is_weights
            sup_loss = torch.sum(sup_loss)
        else:
            sup_loss = None
        
        losses = {"vs_loss": vs_loss,
                  "logits_loss": logits_loss,
                  "sup_loss": sup_loss}        
        total_loss = self.flags.model_vs_loss_cost * vs_loss + self.flags.model_logits_loss_cost * logits_loss
        if self.model_net.pred_net.predict_rd: 
            total_loss = total_loss + self.flags.model_rs_loss_cost * rs_loss
            losses["rs_loss"] = rs_loss
            if self.flags.model_done_loss_cost > 0.:
                total_loss = total_loss + self.flags.model_done_loss_cost * done_loss
                losses["done_loss"] = done_loss
        if self.flags.model_sup_loss_cost > 0.:
            total_loss = total_loss +  self.flags.model_sup_loss_cost * sup_loss

        losses["total_loss_p"] = total_loss

        # compute priorities
        if self.flags.priority_alpha > 0.:
            if not self.flags.model_batch_mode:            
                priorities = torch.absolute(vs - target["vs"])
                if self.flags.priority_type in [1, 2] and not self.flags.perfect_model:    
                    # when the model is imperfect, we only reset the priority of the first time step
                    if self.flags.priority_type == 1:
                        priorities[0] = torch.mean(priorities, dim=0)
                    priorities[1:] = torch.nan                 
                priorities = priorities.detach().cpu().numpy()
            else:
                priorities = (torch.absolute(vs[0] - target["vs"][0])).detach().cpu().numpy()
        else:
            priorities = None

        return losses, priorities

    def prepare_data(self, train_model_out):
        k, b = self.flags.model_k_step_return, train_model_out.gym_env_out.shape[1]
        target_xs = train_model_out.gym_env_out.float() / 255.
        target_rewards = train_model_out.reward[1:k+1]  # true reward r_1, r_2, ..., r_k    
        target_logits = train_model_out.policy_logits[1:k+1] # true logits l_0, l_1, ..., l_k-1
        target_vs = train_model_out.baseline[k:k+k] # baseline ranges from v_k, v_k+1, ... v_2k
        for t in range(k, 0, -1):
            target_vs = target_vs * self.flags.discounting * (~train_model_out.done[t:k+t]).float() + train_model_out.reward[t:k+t]            
            t_done = train_model_out.truncated_done[t:k+t]
            if torch.any(t_done):
                target_vs[t_done] = train_model_out.baseline[t-1:k+t-1][t_done]
        
        # if done on step j, r_j, v_j-1, a_j-1 has the last valid loss 
        # we set all target r_j+1, v_j, a_j to 0, 0, and last a_{j+1} 
        
        if not self.flags.perfect_model:
            trun_done = torch.zeros(k+1, b, dtype=torch.bool, device=self.device) 
            true_done = torch.zeros(k+1, b, dtype=torch.bool, device=self.device) 
            # done_mask stores accumulated done: True, adone_1, adone_2, ..., adone_k
            for t in range(1, k+1):             
                trun_done[t] = torch.logical_or(trun_done[t-1], train_model_out.truncated_done[t])
                true_done[t] = torch.logical_or(true_done[t-1], train_model_out.done[t])
                if not self.flags.model_done_loss_cost > 0.:  target_xs[t, true_done[t]] = 0.                 
                if t < k:
                    target_rewards[t, true_done[t]] = 0.
                    target_logits[t, true_done[t]] = target_logits[t-1, true_done[t]]
                    target_vs[t, true_done[t]] = 0.                     
            if self.flags.model_done_loss_cost > 0.:
                done_mask = (~torch.logical_or(trun_done,  true_done)).float()
                target_done =  torch.logical_and(~trun_done,  true_done).float()[1:]
            else:
                done_mask = (~trun_done).float()              
                target_done = None
        else:
            done_mask = torch.ones(k+1, b, device=self.device) 
            trun_done = None
            target_done = None

        if self.flags.value_prefix:
            target_rewards = torch.cumsum(target_rewards, dim=0)

        return {"xs": target_xs[1:k+1],
                "rewards": target_rewards,
                "logits": target_logits,
                "vs": target_vs,
                "dones": target_done,
                "trun_done": trun_done,
                "done_mask": done_mask,
            }
    
    def gradient_step(self, loss, optimizer, scheduler):
        # gradient descent on loss
        optimizer.zero_grad()
        if self.flags.float16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()        
        optimize_params = optimizer.param_groups[0]['params']
        if self.flags.float16: self.scaler.unscale_(optimizer)
        if self.flags.model_grad_norm_clipping > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(optimize_params, self.flags.model_grad_norm_clipping)
        else:
            total_norm = util.compute_grad_norm(optimize_params)     
        
        if self.flags.float16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            optimizer.step()            
        scheduler.last_epoch = self.real_step - 1  # scheduler does not support setting epoch directly
        scheduler.step()   
        optimizer.zero_grad(set_to_none=True)
        return total_norm              

    def step_per_transition(self):
        return self.step / (self.real_step - self.flags.model_warm_up_n + 1) 

    def save_checkpoint(self):
        basepath = os.path.split(self.check_point_path)[0]
        if not os.path.exists(basepath):
            self._logger.info("Creating log directory: %s" % basepath)
            os.makedirs(basepath, exist_ok=True)

        self._logger.info("Saving model checkpoint to %s" % self.check_point_path)
        d = { "step": self.step,
              "real_step": self.real_step,
              "model_net_optimizer_p_state_dict": self.optimizer_p.state_dict(),
              "model_net_scheduler_p_state_dict": self.scheduler_p.state_dict(),
              "model_net_state_dict": self.model_net.state_dict(),
              "flags":  vars(self.flags)
            }        
        if self.flags.duel_net:
            d.update({
              "model_net_optimizer_m_state_dict": self.optimizer_m.state_dict(),
              "model_net_scheduler_m_state_dict": self.scheduler_m.state_dict(),
              })
        torch.save(d, self.check_point_path)

    def load_checkpoint(self, check_point_path: str):
        train_checkpoint = torch.load(check_point_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        if self.flags.duel_net:
            self.optimizer_m.load_state_dict(train_checkpoint["model_net_optimizer_m_state_dict"])         
            self.scheduler_m.load_state_dict(train_checkpoint["model_net_scheduler_m_state_dict"])       
        self.optimizer_p.load_state_dict(train_checkpoint["model_net_optimizer_p_state_dict"])         
        self.scheduler_p.load_state_dict(train_checkpoint["model_net_scheduler_p_state_dict"])                   
        self.model_net.set_weights(train_checkpoint["model_net_state_dict"])        
        self._logger.info("Loaded model checkpoint from %s" % check_point_path)  

    def close(self, exit_code):
        self.plogger.close()
        if self.flags.use_wandb: self.wlogger.wandb.finish(exit_code=exit_code)