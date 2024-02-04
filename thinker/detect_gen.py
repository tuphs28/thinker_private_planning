from thinker.util import __project__
import os
import argparse
import yaml
import numpy as np
import torch
from thinker.actor_net import ActorNet
from thinker.main import Env
import thinker.util as util
from thinker.self_play import init_env_out, create_env_out


class DetectBuffer:
    def __init__(self, outdir, t, rec_t, logger, delay_n=5):
        """
        Store training data grouped in planning stages and output
        whenever the target output is also readydd
            Args:
                N (int): number of planning stage per training output
                delay_n (int): number of planning stage delayed in the output y
                rec_t (int): number of step in a planning stage
                K (int): number of block to merge into
        """
        self.outdir = outdir
        self.t = t # number of time step per file
        self.rec_t = rec_t
        self.logger = logger        
        self.delay_n = delay_n        

        self.processed_n, self.xs, self.y, self.done, self.step_status = 0, [], [], [], []
        self.file_idx = -1
    
    def insert(self, xs, y, done, step_status):
        """
        Args:
            xs (dict): dictionary of training input, with each elem having the
                shape of (B, *)            
            y (tensor): bool tensor of shape (B), being the target output delayed by
                delay_n planning stage            
            done (tensor): bool tensor of shape (B), being the indicator of episode end
            step_status (int): int indicating current step status
        Output:
            save train_xs in shape (N, rec_t, B, *) and train_y in shape (N, B)
        """
        #print("data received! ", y.shape, id, cur_t)
        last_step_real = (step_status == 0) | (step_status == 3)
        if len(self.step_status) == 0 and not last_step_real: return self.file_idx  # skip until real step
                
        self.xs.append(util.dict_map(xs, lambda x:x.cpu()))
        self.y.append(y.cpu())
        self.done.append(done.cpu())
        self.step_status.append(step_status)
        self.processed_n += int(last_step_real)

        if (self.processed_n >= self.t + self.delay_n + 1):               
            self.file_idx += 1                     
            out = self._extract_data(self.t)
            self.processed_n = sum([int(i == 0) + int(i == 3) for i in self.step_status])
            assert self.processed_n == self.delay_n+1, f"should only have {self.delay_n + 1} data left instead of {self.processed_n}"
            path = f'{self.outdir}/data_{self.file_idx}.pt'
            torch.save(out, path)
            out_shape = out[0]['env_state'].shape
            n = self.file_idx * out_shape[0] * out_shape[2]
            self.logger.info(f"{n}: File saved to {path}; env_state shape {out_shape}")

        return self.file_idx   

    def _extract_data(self, t):
        # obtain the first N planning stage and the corresponding target_y in data
        xs, y, done, step_status = self._collect_data(t)
        future_y, future_done = self._collect_data(self.delay_n, y_done_only=True)
        y = torch.concat([y, future_y], dim=0)
        done = torch.concat([done, future_done], dim=0)                
        
        last_step_real = (step_status == 0) | (step_status == 3)
        assert last_step_real[0], "cur_t should start with 0"
        assert last_step_real.shape[0] == t*self.rec_t, \
            f" step_status.shape is {last_step_real.shape}, expected {t*self.rec_t} for the first dimension."        
        assert y.shape[0] == (t + self.delay_n)*self.rec_t, \
            f" y.shape is {y.shape}, expected {(t + self.delay_n)*self.rec_t} for the first dimension."        
        
        B = y.shape[1]
        y = y.view(t + self.delay_n, self.rec_t, B)[:, 0]
        done = done.view(t + self.delay_n, self.rec_t, B)[:, 0]
        step_status = step_status.view(t, self.rec_t)
        # compute target_y
        target_y = self._compute_target_y(y, done, self.delay_n)

        for k in xs.keys():
            xs[k] = xs[k].view((t, self.rec_t) + xs[k].shape[1:])
        
        xs["done"] = done[:t]
        xs["step_status"] = step_status
                
        return xs, target_y

    def _collect_data(self, t, y_done_only=False):
        # collect the first t stage from data
        step_status = torch.tensor(self.step_status, dtype=torch.long)
        next_step_real = (step_status == 2) | (step_status == 3)        
        idx = torch.nonzero(next_step_real, as_tuple=False).squeeze()    
        last_idx = idx[t-1] + 1
        y = torch.stack(self.y[:last_idx], dim=0)
        done = torch.stack(self.done[:last_idx], dim=0)
        if not y_done_only:
            xs = {}
            for k in self.xs[0].keys():
                xs[k] = torch.stack([v[k] for v in self.xs[:last_idx]], dim=0)                
            step_status = step_status[:last_idx]
            self.xs = self.xs[last_idx:]
            self.y = self.y[last_idx:]
            self.done = self.done[last_idx:]
            self.step_status = self.step_status[last_idx:]
            return xs, y, done, step_status
        else:
            return y, done
        
    def _compute_target_y(self, y, done, delay_n):        
        # target_y[i] = (y[i] | (~done[i+1] & y[i+1]) | (~done[i+1] & ~done[i+2] & y[i+2]) | ... | (~done[i+1] & ~done[i+2] & ... & ~done[i+M] & y[i+M]))
        t, b = y.shape
        t = t - delay_n
        not_done_cum = torch.ones(delay_n, t, b, dtype=bool)
        target_y = y.clone()[:-delay_n]
        not_done_cum[0] = ~done[1:1+t]
        target_y = target_y | (not_done_cum[0] & y[1:1+t])
        for m in range(1, delay_n):
            not_done_cum[m] = not_done_cum[m-1] & ~done[m+1:m+1+t]
            target_y = target_y | (not_done_cum[m] & y[m+1:m+1+t])
        return target_y

def detect_gen(total_n, env_n, delay_n, greedy, savedir, outdir, xpid):

    _logger = util.logger()
    _logger.info(f"Initializing {xpid} from {savedir}")
    device = torch.device("cuda")

    ckpdir = os.path.join(savedir, xpid)     
    if os.path.islink(ckpdir): ckpdir = os.readlink(ckpdir)  
    ckpdir =  os.path.abspath(os.path.expanduser(ckpdir))
    outdir = os.path.abspath(os.path.expanduser(outdir))

    config_path = os.path.join(ckpdir, 'config_c.yaml')
    flags = util.create_flags(config_path, save_flags=False)
    disable_thinker = flags.wrapper_type == 1

    env = Env(
            name=flags.name,
            env_n=env_n,
            gpu=True,
            train_model=False,
            parallel=False,
            savedir=savedir,        
            xpid=xpid,
            ckp=True,
            return_x=True,
            return_h=True,
        )

    obs_space = env.observation_space
    action_space = env.action_space 

    actor_param = {
        "obs_space": obs_space,
        "action_space": action_space,
        "flags": flags,
        "tree_rep_meaning": env.get_tree_rep_meaning() if not disable_thinker else None,
        "record_state": True,
    }
    actor_net = ActorNet(**actor_param)

    path = os.path.join(ckpdir, "ckp_actor.tar")
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    actor_net.set_weights(checkpoint["actor_net_state_dict"])
    actor_net.to(device)
    actor_net.train(False)

    state = env.reset()
    env_out = init_env_out(state, flags=flags, dim_actions=actor_net.dim_actions, tuple_action=actor_net.tuple_action)  
    actor_state = actor_net.initial_state(batch_size=env_n, device=device)

    # create dir

    n = 0
    while True:
        name = "%s-%d-%d" % (xpid, checkpoint["real_step"], n)
        outdir_ = os.path.join(outdir, name)
        if not os.path.exists(outdir_):
            os.makedirs(outdir_)
            print(f"Outputting to {outdir_}")
            break
        n += 1
    outdir = outdir_

    detect_buffer = DetectBuffer(outdir=outdir, t=12800//env_n, rec_t=flags.rec_t, logger=_logger, delay_n=delay_n)
    file_n = total_n // (env_n * detect_buffer.t) + 1
    _logger.info(f"Data output directory: {outdir}")
    _logger.info(f"Number of file to be generated: {file_n}")

    rescale = "Sokoban" in flags.name

    
    with torch.set_grad_enabled(False):
        
        actor_out, actor_state = actor_net(env_out=env_out, core_state=actor_state, greedy=greedy)            
        if not disable_thinker:
            primary_action, reset_action = actor_out.action
        else:
            primary_action, reset_action = actor_out.action, None

        # save setting
        env_state_shape = env.observation_space["real_states"].shape[1:]
        #if rescale: env_state_shape = (3, 40, 40)
        tree_rep_shape = env.observation_space["tree_reps"].shape[1:] if not disable_thinker else None
        hidden_state_shape = actor_net.hidden_state.shape[1:] if disable_thinker else None

        flags_detect = {
            "dim_actions": actor_net.dim_actions,
            "num_actions": actor_net.num_actions,
            "tuple_actions": actor_net.tuple_action,
            "name": flags.name,
            "env_state_shape": list(env_state_shape),
            "tree_rep_shape": list(tree_rep_shape) if not disable_thinker else None,
            "hidden_state_shape": list(hidden_state_shape) if disable_thinker else None,
            "rescale": rescale,
            "rec_t": flags.rec_t,
            "ckpdir": ckpdir,
            "xpid": xpid,        
            "dxpid": name,
            "disable_thinker": disable_thinker,
        }

        yaml_file_path = os.path.join(outdir, 'config_detect.yaml')
        with open(yaml_file_path, 'w') as file:
            yaml.dump(flags_detect, file)


        rets = []
        last_file_idx = None
        
        while(True):
            state, reward, done, info = env.step(
                primary_action=primary_action, 
                reset_action=reset_action, 
                action_prob=actor_out.action_prob[-1])    
            
            env_out = create_env_out(actor_out.action, state, reward, done, info, flags=flags)
            if torch.any(done):
                rets.extend(info["episode_return"][done].cpu().tolist())

            actor_out, actor_state = actor_net(env_out=env_out, core_state=actor_state, greedy=greedy)            
            if not disable_thinker:
                primary_action, reset_action = actor_out.action
            else:
                primary_action, reset_action = actor_out.action, None
            
            # write to detect buffer
            if not disable_thinker:
                env_state = env_out.xs[0] 
                if rescale:
                    #env_state = F.interpolate(env_state , size=(40, 40), mode='bilinear', align_corners=False)
                    env_state = (env_state * 255).to(torch.uint8)
            else:
                env_state = env_out.real_states[0]
            xs = {
                "env_state": env_state,
                "pri_action": primary_action,            
                "cost": info["cost"],
            }
            if not disable_thinker:
                xs.update({
                    "tree_rep": state["tree_reps"],
                    "reset_action": actor_out.action[1],
                })
            else:
                if flags.drc:
                    xs.update({
                        "hidden_state": actor_net.hidden_state
                    })       
            y = info['cost']
            done = done
            step_status = info['step_status'][0].item()

            file_idx = detect_buffer.insert(xs, y, done, step_status)
            
            if file_idx >= file_n: 
                # last file is for validation
                os.rename(f'{outdir}/data_{file_idx}.pt', f'{outdir}/val.pt')
                break

            if last_file_idx is not None and file_idx != last_file_idx:
                print(f"Episode {len(rets)}; Return  {np.mean(np.array(rets))}")

            last_file_idx = file_idx
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Thinker data generalization")
    parser.add_argument("--outdir", default="../data/__project__", help="Output directory.")
    parser.add_argument("--savedir", default="../logs/__project__", help="Checkpoint directory.")
    parser.add_argument("--xpid", default="latest", help="id of the run.")    
    parser.add_argument("--project", default="", help="project of the run.")  
    parser.add_argument("--total_n", default=100000, type=int, help="Number of real steps.")
    parser.add_argument("--env_n", default=128, type=int, help="Batch size in generation.")
    parser.add_argument("--delay_n", default=5, type=int, help="Delay step in predicting danger.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy policy.")

    flags = parser.parse_args()    
    project = flags.project if flags.project else __project__
    
    flags.outdir=flags.outdir.replace("__project__", project)
    flags.savedir=flags.savedir.replace("__project__", project)

    detect_gen(
        total_n=flags.total_n,
        env_n=flags.env_n,
        delay_n=flags.env_n,
        greedy=flags.greedy,
        savedir=flags.savedir,
        outdir=flags.outdir,
        xpid=flags.xpid,
    )