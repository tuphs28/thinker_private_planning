import os
import numpy as np
import argparse
import time
import timeit
import ray
import torch
import torch.nn.functional as F
from thinker.core.file_writer import FileWriter
from thinker.buffer import GeneralBuffer, ModelBuffer
from thinker.net import ModelNet
from thinker.self_play import SelfPlayWorker, TrainModelOut, PO_MODEL
from thinker.env import Environment
import thinker.util as util

def compute_cross_entropy_loss(logits, target_logits, is_weights):
    target_policy = F.softmax(target_logits, dim=-1)
    loss = torch.nn.CrossEntropyLoss(reduction="none")(
                    input = torch.flatten(logits, 0, 1),
                    target = torch.flatten(target_policy, 0, 1)
                )
    loss = torch.sum(loss.reshape(target_logits.shape[:-1]), dim=0)
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
        
        self.optimizer = torch.optim.Adam(self.model_net.parameters(),lr=flags.model_learning_rate)
        lr_lambda = lambda epoch: 1 - min(epoch, self.flags.total_steps) / self.flags.total_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.flags.preload_model and not flags.load_checkpoint:
            checkpoint = torch.load(self.flags.preload_model, map_location=torch.device('cpu'))
            self.model_net.set_weights(checkpoint["model_state_dict" if "model_state_dict" in checkpoint else "model_net_state_dict"])  
            self._logger.info("Loadded model network from %s" % self.flags.preload_model)
            if "model_net_optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["model_net_optimizer_state_dict"])  
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
        util.optimizer_to(self.optimizer, self.device)

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
            flags=flags) for n in range(5)]

    def learn_data(self):
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
            scaler = torch.cuda.amp.GradScaler(init_scale=2**8)

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
                        
            # compute losses
            if self.flags.float16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):      
                    losses, priorities = self.compute_losses(train_model_out, is_weights)
            else:
                losses, priorities = self.compute_losses(train_model_out, is_weights)
            total_loss = losses["total_loss"]
            self.model_buffer.update_priority.remote(inds, priorities)
            
            # gradient descent on loss
            self.optimizer.zero_grad()
            if self.flags.float16:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            optimize_params = self.optimizer.param_groups[0]['params']
            if self.flags.float16: scaler.unscale_(self.optimizer)
            if self.flags.model_grad_norm_clipping > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(optimize_params, self.flags.model_grad_norm_clipping)
            else:
                total_norm = util.compute_grad_norm(optimize_params)     
            
            if self.flags.float16:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
                
            self.scheduler.last_epoch = self.real_step - 1  # scheduler does not support setting epoch directly
            self.scheduler.step()                 
            
            self.step += numel_per_step
            # print statistics
            if timer() - start_time > 5:
                sps = (self.step - start_step) / (timer() - start_time)                
                print_str =  "Steps %i (%i:%i[%.1f]) @ %.1f SPS. Model return mean (std) %f (%f) total_norm %.2f" % (
                                n, 
                                self.real_step, 
                                self.step, 
                                self.step_per_transition(), 
                                sps, 
                                np.mean(all_returns) if all_returns is not None else 0.,
                                np.std(all_returns) if all_returns is not None else 0.,
                                total_norm.item())
                print_stats = ["total_loss", "vs_loss", "logits_loss", "rs_loss", "sup_loss"]
                for k in print_stats: 
                    if losses[k] is not None:
                        print_str += " %s %.2f" % (k, losses[k].item() / numel_per_step)
                self._logger.info(print_str)
                start_step = self.step
                start_time = timer()      

                # write to log file
                stats = {"step": self.step,
                         "real_step": self.real_step,
                         "model_returns_mean": np.mean(all_returns) if all_returns is not None else None,
                         "model_returns_std": np.std(all_returns)/np.sqrt(len(all_returns)) if all_returns is not None else None,
                         "model_total_norm": total_norm.item()}
                for k in print_stats: 
                    stats[k] = losses[k].item() / numel_per_step if losses[k] is not None else None

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
            if self.step - start_step_test > 500000 * self.flags.rec_t:
                start_step_test = self.step
                if r_tester is not None: 
                    all_returns = ray.get(r_tester)[0]
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
        return True

    def compute_losses(self, train_model_out: TrainModelOut, is_weights: torch.Tensor):
        k, b = self.flags.model_k_step_return, train_model_out.gym_env_out.shape[1]
        train_len = train_model_out.gym_env_out.shape[0] - k
        # each elem in train_model_out is in the shape of (train_len + k, b, ...)
    
        gym_env_out = train_model_out.gym_env_out
        if self.flags.perfect_model: 
            gym_env_out_ = gym_env_out[:train_len] # the last k elem are not needed         
            action = train_model_out.action[:train_len]
            _, _, vs, v_enc_logits, logits, _ = self.model_net(
                x=gym_env_out_.reshape((-1,) + train_model_out.gym_env_out.shape[2:]),
                actions=action.reshape(1, -1),  one_hot=False)
            vs = vs.reshape(k, b)
            if v_enc_logits is not None: v_enc_logits = v_enc_logits.reshape(k, b, -1)
            logits = logits.reshape(k, b, -1)
        else:        
            action = train_model_out.action[:train_len+1]
            rs, r_enc_logits, vs, v_enc_logits, logits, encodeds = self.model_net(
                x=gym_env_out[0], actions=action, one_hot=False)
            vs = vs[:-1]
            v_enc_logits = v_enc_logits[:-1]
            logits = logits[:-1]

        target_rewards = train_model_out.reward[1:train_len+1]
        target_logits = train_model_out.policy_logits[1:train_len+1]

        target_vs = train_model_out.baseline[k:train_len+k]
        for t in range(k, 0, -1):
            target_vs = target_vs * self.flags.discounting * (~train_model_out.done[t:train_len+t]).float() + train_model_out.reward[t:train_len+t]

        # if done on step j, r_{j}, v_{j-1}, a_{j-1} has the last valid loss 
        # we set all target r_{j+1}, v_{j}, a_{j} to 0, 0, and last a_{j+1}
        # rs is stored in the form of r_{t+1}, ..., r_{t+k}
        # vs is stored in the form of v_{t}, ..., v_{t+k-1}
        # logits is stored in the form of a{t}, ..., a_{t+k-1}        

        if not self.flags.perfect_model:
            done = torch.zeros(b, dtype=torch.bool, device=self.device)
            for t in range(train_len-1):
                done = torch.logical_or(done, train_model_out.done[t+1])
                target_rewards[t+1, done] = 0.
                target_logits[t+1, done] = target_logits[t, done]
                target_vs[t+1, done] = 0.
                if self.flags.model_supervise:
                    gym_env_out[t+1, done] = gym_env_out[t, done]    
        
        # compute final loss

        #huberloss = torch.nn.HuberLoss(reduction='none', delta=1.0)    
        #rs_loss = torch.sum(huberloss(rs, target_rewards.detach()) * (~done_masks).float())
        if self.flags.perfect_model:
            rs_loss =  None
        else:
            if not self.flags.reward_transform:
                rs_loss = torch.sum(((rs - target_rewards) ** 2), dim=0)
            else:
                _, target_rs_enc_v = self.model_net.reward_tran.encode(target_rewards)
                rs_loss = torch.nn.CrossEntropyLoss(reduction="none")(
                        input = torch.flatten(r_enc_logits[:train_len], 0, 1),
                        target = torch.flatten(target_rs_enc_v, 0, 1)
                      )
                rs_loss = torch.sum(rs_loss.reshape(target_rs_enc_v.shape[:-1]), dim=0)
            rs_loss = rs_loss * is_weights
            rs_loss = torch.sum(rs_loss)
            
        #vs_loss = torch.sum(huberloss(vs[:-1], target_vs.detach()))
        if not self.flags.reward_transform:
            vs_loss = torch.sum(((vs[:train_len] - target_vs.detach()) ** 2), dim=0)
        else:
            _, target_vs_enc_v = self.model_net.reward_tran.encode(target_vs)
            vs_loss = torch.nn.CrossEntropyLoss(reduction="none")(
                        input = torch.flatten(v_enc_logits[:train_len], 0, 1),
                        target = torch.flatten(target_vs_enc_v.detach(), 0, 1)
                      )
            vs_loss = torch.sum(vs_loss.reshape(target_vs_enc_v.shape[:-1]), dim=0)

        vs_loss = vs_loss * is_weights
        vs_loss = torch.sum(vs_loss)

        logits_loss = compute_cross_entropy_loss(
            logits, target_logits.detach(), is_weights)   

        if self.flags.model_supervise:
            sup_loss = self.model_net.supervise_loss(encodeds=encodeds[1:], x=gym_env_out[1:train_len+1], 
                actions=train_model_out.action[1:train_len+1], is_weights=is_weights, one_hot=False)
        else:
            sup_loss = None

        total_loss = self.flags.model_vs_loss_cost * vs_loss + self.flags.model_logits_loss_cost * logits_loss
        if rs_loss is not None: total_loss = total_loss + self.flags.model_rs_loss_cost * rs_loss
        if sup_loss is not None: total_loss = total_loss + self.flags.model_sup_loss_cost * sup_loss

        losses = {"total_loss": total_loss,
                   "vs_loss": vs_loss,
                   "logits_loss": logits_loss,
                   "rs_loss": rs_loss,
                   "sup_loss": sup_loss }

        # compute priorities
        if not self.flags.model_batch_mode:            
            priorities = torch.absolute(vs - target_vs)
            if self.flags.priority_type in [1, 2] and not self.flags.perfect_model:    
                # when he model is imperfect, we only reset the priority of the first time step
                if self.flags.priority_type == 1:
                    priorities[0] = torch.mean(priorities, dim=0)
                priorities[1:] = torch.nan                 
            priorities = priorities.detach().cpu().numpy()
        else:
            priorities = (torch.absolute(vs[0] - target_vs[0])).detach().cpu().numpy()
        return losses, priorities

    def step_per_transition(self):
        return self.step / (self.real_step - self.flags.model_warm_up_n + 1) 

    def save_checkpoint(self):
        basepath = os.path.split(self.check_point_path)[0]
        if not os.path.exists(basepath):
            self._logger.info("Creating log directory: %s" % basepath)
            os.makedirs(basepath, exist_ok=True)

        self._logger.info("Saving model checkpoint to %s" % self.check_point_path)
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
        self._logger.info("Loaded model checkpoint from %s" % check_point_path)   
