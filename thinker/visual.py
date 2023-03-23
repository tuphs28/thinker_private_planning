import os
import sys
from collections import deque
import time
import numpy as np
import copy
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn.functional as F
from thinker.env import Environment
from thinker.net import ActorNet, ModelNet
import thinker.util as util
import gym

def plot_gym_env_out(x, ax=None, title=None):
    if ax is None: fig, ax = plt.subplots()
    ax.imshow(torch.swapaxes(torch.swapaxes(x[0,-3:].cpu(),0,2),0,1), interpolation='nearest', aspect="auto")
    if title is not None: ax.set_title(title)

def plot_multi_gym_env_out(xs, titles=None, col_n=5):
    size_n = 6
    row_n = (len(xs) + (col_n-1))// col_n
    
    fig, axs = plt.subplots(row_n, col_n, figsize=(col_n * size_n, row_n * size_n))  
    if(len(axs.shape) == 1): axs = axs[np.newaxis, :]
    m = 0
    for y in range(row_n):
        for x in range(col_n):
            if m >= len(xs): 
                axs[y][x].set_axis_off()
            else:
                axs[y][x].imshow(np.transpose(xs[m][-3:], axes=(1, 2, 0))/255)
                axs[y][x].set_title("rollout %d" % (m+1) if titles is None else titles[m])
            m += 1
    plt.tight_layout()
    return fig

def plot_policies(logits, labels, action_meanings, ax=None, title="Real policy prob"):
    if ax is None: fig, ax = plt.subplots()
    probs = []
    for logit, k in zip(logits, labels):
        if k != "action":
            probs.append(torch.softmax(logit, dim=-1).detach().cpu().numpy())
        else:
            probs.append(logit.detach().cpu().numpy())

    ax.set_title(title)
    xs = np.arange(len(probs[0]))    
    for n, (prob, label) in enumerate(zip(probs, labels)):
        ax.bar(xs + 0.1 * (n-len(logits) // 2), prob, width = 0.1, label=label)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(probs[0]))))
    ax.set_xticklabels(action_meanings)
    ax.set_ylim(0, 1)        
    ax.legend()        

def plot_base_policies(logits, action_meanings, ax=None):
    if ax is None: fig, ax = plt.subplots()
    prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    rec_t, num_actions = logits.shape
    xs = np.arange(rec_t)
    labels = action_meanings
    for i in range(num_actions):        
        c = ax.bar(xs + 0.8 * (i / num_actions), prob[:,i], width = 0.8 / (num_actions), label=labels[i])  
        color = c.patches[0].get_facecolor()
        color = color[:3] + (color[3] * 0.5,)
        ax.bar(xs + 0.8 * (i / num_actions), prob[:,i], width = 0.8 / (num_actions), color=color)
    ax.legend()
    ax.set_ylim(0, 1)   
    ax.set_title("Model policy prob")

def plot_im_policies(im_policy_logits, reset_policy_logits,  
                     im_action, reset_action, action_meanings,
                     one_hot=True, reset_ind=0, ax=None):
    if ax is None: fig, ax = plt.subplots()
        
    rec_t, num_actions = im_policy_logits.shape
    num_actions += 1
    rec_t -= 1
        
    im_prob = torch.softmax(im_policy_logits, dim=-1).detach().cpu().numpy()
    reset_prob = torch.softmax(reset_policy_logits, dim=-1)[:,[reset_ind]].detach().cpu().numpy()
    full_prob = np.concatenate([im_prob, reset_prob], axis=-1)
    
    if not one_hot: im_action = F.one_hot(im_action, num_actions - 1)
    im_action = im_action.detach().cpu().numpy()
    reset_action = reset_action.unsqueeze(-1).detach().cpu().numpy()    
    full_action = np.concatenate([im_action, reset_action], axis=-1)
    
    xs = np.arange(rec_t+1)
    labels = action_meanings.copy()
    labels.append('RESET')    
        
    for i in range(num_actions):        
        c = ax.bar(xs + 0.8 * (i / num_actions), full_prob[:,i], width = 0.8 / (num_actions), label=labels[i])  
        color = c.patches[0].get_facecolor()
        color = color[:3] + (color[3] * 0.5,)
        ax.bar(xs + 0.8 * (i / num_actions), full_action[:,i], width = 0.8 / (num_actions), color=color)
        
    ax.legend()
    ax.set_ylim(0, 1)   
    ax.set_title("Imagainary policy prob")    

def plot_qn_sa(q_s_a, n_s_a, action_meanings, max_q_s_a=None, ax=None):
    if ax is None: fig, ax = plt.subplots()
    xs = np.arange(len(q_s_a))

    ax.bar(xs - 0.3, q_s_a.cpu(), color = 'g', width = 0.3, label="q_s_a")    
    ax_n = ax.twinx()
    if max_q_s_a is not None:
        ax.bar(xs, max_q_s_a.cpu(), color = 'r', width = 0.3, label="max_q_s_a")        
    ax_n.bar(xs + (0.3 if max_q_s_a is not None else 0.), 
             n_s_a.cpu(), bottom=0, color = 'b', width = 0.3, label="n_s_a")
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(q_s_a))))
    ax.set_xticklabels(action_meanings)    
    ax.legend(loc="lower left")   
    ax_n.legend(loc="lower right") 
    ax.set_title("q_s_a and n_s_a")    

def gen_video(video_stats, file_path):
    # Generate video
    import cv2
    imgs = []
    hw = video_stats["real_imgs"][0].shape[1]

    for i in range(len(video_stats["real_imgs"])):
        img = np.zeros(shape=(hw,hw*2,3),dtype=np.uint8)
        real_img = np.copy(video_stats["real_imgs"][i])
        real_img = np.swapaxes(np.swapaxes(real_img, 0, 2),0, 1)
        im_img = np.copy(video_stats["im_imgs"][i])
        im_img = np.swapaxes(np.swapaxes(im_img, 0, 2),0, 1)
        if video_stats["status"][i] == 1: 
            im_img[:, :, 0] = 255 * 0.3 + im_img[:, :, 0] * 0.7
            im_img[:, :, 1] = 255 * 0.3 + im_img[:, :, 1] * 0.7
        elif video_stats["status"][i] == 0: 
            im_img[:, :, 2] = 255 * 0.3 + im_img[:, :, 2] * 0.7
        img[:,:hw,:] = real_img
        img[:,hw:,:] = im_img
        img = np.flip(img, 2)
        imgs.append(img)

    width = hw*2
    hieght = hw    
    fps = 15
    
    video = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, hieght)) 
    for img in imgs: video.write(img) 
    video.release()        

def print_im_actions(im_dict, action_meanings, print_stat=False):
    lookup_dict = {k:v for k, v in enumerate(action_meanings)}
    print_strs = []
    n, s = 1, ""
    reset = False
    for im, reset in zip(im_dict["im_action"][:-1], im_dict["reset_action"][:-1]):
        s += lookup_dict[im.item()] + ", "
        if reset:        
            s += "Reset"
            print_strs.append("%d: %s" %(n, s))
            s = ""
            n += 1
    if not reset: print_strs.append("%d: %s" %(n, s[:-2]))
    if print_stat: 
        for s in print_strs: print(s) 
    return print_strs    

def visualize(check_point_path, model_path="", visualize=False, saveimg=True, savevideo=True):
    max_eps_n = 1
    saveimg_dir = "/home/sc/RS/thinker/test/data/"

    flags = util.parse(['--load_checkpoint', check_point_path])
    #flags.actor_net_ver = 0
    name = "%s-%s"%(flags.xpid, time.strftime("%Y%m%d-%H%M%S"))

    if saveimg:
        saveimg_a = os.path.join(saveimg_dir, name, "a")
        saveimg_b = os.path.join(saveimg_dir, name, "b")
        if not (os.path.exists(saveimg_a)): os.makedirs(saveimg_a)
        if not (os.path.exists(saveimg_b)): os.makedirs(saveimg_b)
        print("saving images to %s" % (os.path.join(saveimg_dir, name)))
        savef = open(os.path.join(saveimg_dir, name, "logs.txt"), 'a')
    env = Environment(flags, model_wrap=True)
    if not flags.perfect_model: 
        flags_ = copy.deepcopy(flags)
        flags_.perfect_model = True
        flags_.actor_see_p = 1
        perfect_env = Environment(flags_, model_wrap=True)

    if flags.env == "cSokoban-v0" or flags.env == "Sokoban-v0":
        action_meanings = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT"]
    else:
        action_meanings = gym.make(flags.env).get_action_meanings()
    num_actions = env.num_actions

    env.seed([0])
    if not flags.perfect_model: perfect_env.seed([0])

    model_net = ModelNet(obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags)
    model_net.train(False)
    if not model_path: model_path = os.path.join(check_point_path,'ckp_model.tar')
    checkpoint = torch.load(model_path, torch.device("cpu"))
    model_net.set_weights(checkpoint["model_state_dict"] if "model_state_dict" in 
                        checkpoint else checkpoint["model_net_state_dict"])  

    actor_net = ActorNet(obs_shape=env.model_out_shape, gym_obs_shape=env.gym_env_out_shape, num_actions=env.num_actions, flags=flags)
    checkpoint = torch.load(os.path.join(check_point_path,'ckp_actor.tar'), torch.device("cpu"))         
    actor_net.set_weights(checkpoint["actor_net_state_dict"])
    actor_state = actor_net.initial_state(batch_size=1) 

    env_out = env.initial(model_net)
    if not flags.perfect_model: 
        perfect_env_out = perfect_env.initial(model_net)
        assert torch.all(env_out.gym_env_out == perfect_env_out.gym_env_out)

    gym_env_out_ = env_out.gym_env_out
    model_out = util.decode_model_out(env_out.model_out, num_actions, flags.reward_transform)
    end_gym_env_outs, end_titles = [], []
    ini_max_q = model_out["root_max_v"][0].item()

    step = 0
    returns, model_logits, attn_output = [], [], [], 
    im_list = ["im_policy_logits", "reset_policy_logits", "im_action", "reset_action"]
    im_dict = {k: [] for k in im_list}

    video_stats = {"real_imgs": [], "im_imgs":[], "status":[]}
    video_stats["real_imgs"].append(env_out.gym_env_out[0,0,-3:].numpy())
    video_stats["im_imgs"].append(video_stats["real_imgs"][-1])
    video_stats["status"].append(0) # 0 for real step, 1 for reset, 2 for normal

    while len(returns) < max_eps_n:    
        step += 1
        actor_out, actor_state = actor_net(env_out, actor_state)      

        if env_out.cur_t[0,0] == 0: agent_v = actor_out.baseline[0,0,0]
        action = [actor_out.action, actor_out.im_action, actor_out.reset_action]
        action = torch.cat(action, dim=-1).unsqueeze(0)

        # additional stat record    
        for k in im_list: 
            im_dict[k].append(getattr(actor_out, k)[:, 0] if k in actor_out._fields and 
                getattr(actor_out, k) is not None else None)        
        #attn_output.append(torch.cat([x.attn_output_weights.unsqueeze(0).unsqueeze(-2) for x in actor_net.core.layers])[:, :, 0, :])        

        model_out_ = util.decode_model_out(env_out.model_out, num_actions, flags.reward_transform)
        model_logits.append(model_out_["cur_logits"])
        env_out  = env.step(action, model_net)
        model_out = util.decode_model_out(env_out.model_out, num_actions, flags.reward_transform)
        if len(im_dict['reset_action']) > 0: im_dict['reset_action'][-1] = model_out["reset"]
        if not flags.perfect_model: perfect_env_out  = perfect_env.step(action, model_net)    
        gym_env_out = env_out.gym_env_out if flags.perfect_model else perfect_env_out.gym_env_out
        if not flags.perfect_model:
            perfect_model_out = util.decode_model_out(perfect_env_out.model_out, num_actions, flags.reward_transform)

        if model_out["reset"] == 1 or env_out.cur_t[0,0] == flags.rec_t-1:   
            title = "pred v: %.2f" % (model_out["cur_v"].item())
            if not flags.perfect_model:
                title += " v: %.2f" % (perfect_model_out["cur_v"].item())
            end_gym_env_outs.append(gym_env_out[0,0].numpy())
            end_titles.append(title)

        if not flags.perfect_model and env_out.cur_t[0,0] == 0:
            assert torch.all(env_out.gym_env_out == perfect_env_out.gym_env_out)    

        # record data for generating video
        if action[0,0,2] == 1:
            video_stats["real_imgs"].append(video_stats["real_imgs"][-1])
            video_stats["im_imgs"].append(video_stats["real_imgs"][-1])
            video_stats["status"].append(1)

        if env_out.cur_t[0,0] == 0:
            video_stats["real_imgs"].append(gym_env_out[0,0,-3:].numpy())
            video_stats["status"].append(0)
        else:
            video_stats["real_imgs"].append(video_stats["real_imgs"][-1])
            video_stats["status"].append(2)
        video_stats["im_imgs"].append(gym_env_out[0,0,-3:].numpy())    

        # visualize when a real step is made
        if (saveimg or visualize) and env_out.cur_t[0,0] == 0:
            fig, axs = plt.subplots(1, 5, figsize=(30,6))      
            title = "%d; v: %.2f (%.2f)" % (step, model_out_["root_v"][0], agent_v)
            title += " max_q: %.2f ini_max_q: %.2f mean_q_0: %.2f" % (
                model_out_["root_max_v"][0].item(), ini_max_q, env.env.baseline_mean_q[0])
            for k in im_list: 
                if im_dict[k][0] is not None:
                    im_dict[k] = torch.concat(im_dict[k], dim=0)            
                else:
                    im_dict[k] = None        
            
            if flags.reward_type == 1: title += " im_return: %.4f" % env_out.episode_return[..., 1]                        
            plot_gym_env_out(gym_env_out_[0], axs[0], title=title)  
            plot_base_policies(torch.concat(model_logits), action_meanings=action_meanings, ax=axs[1])  
            plot_im_policies(**im_dict, action_meanings=action_meanings, one_hot=False, reset_ind=1, ax=axs[2])   

            mean_q_s_a = model_out_["root_qs_mean"][0]
            max_q_s_a = model_out_["root_qs_max"][0]
            n_s_a = model_out_["root_ns"][0]
            plot_qn_sa(mean_q_s_a, n_s_a, action_meanings=action_meanings, max_q_s_a=max_q_s_a, ax=axs[3]) 

            model_policy_logits = model_out_["root_logits"][0]
            agent_policy_logits = actor_out.policy_logits[0,0]
            action = torch.nn.functional.one_hot(actor_out.action[0,0], env.num_actions)
            plot_policies([model_policy_logits, agent_policy_logits, action], 
                ["model policy", "agent policy", "action"], action_meanings=action_meanings, ax=axs[4])  
            
            if saveimg: plt.savefig(os.path.join(saveimg_a,"%d.png" %step))
            if visualize: plt.show()        
            plt.close()

            fig = plot_multi_gym_env_out(end_gym_env_outs, end_titles)        
            if saveimg: plt.savefig(os.path.join(saveimg_b,"%d.png" %step))
            if visualize: plt.show()
            plt.close()

            log_str = "step:%d; return %.4f(%.4f) done %s real_done %s" % (
                step, env_out.episode_return[0,0,0], env_out.episode_return[0,0,1] if flags.reward_type == 1 else 0,
                "True" if env_out.done[0,0] else "False", 
                "True" if env_out.real_done[0,0] else "False",)
            print(log_str)
            im_action_strs = print_im_actions(im_dict, action_meanings, print_stat=visualize)           
            if saveimg:
                for s in im_action_strs: savef.write(s+"\n")            
                savef.write(log_str+"\n") 

            gym_env_out_ = gym_env_out
            im_dict = {k: [] for k in im_list}
            model_logits, attn_output, end_gym_env_outs, end_titles = [], [], [], []
            ini_max_q = model_out["root_max_v"][0].item()

        if torch.any(env_out.real_done):
            step = 0
            new_rets = env_out.episode_return[env_out.real_done][:,0].numpy()
            returns.extend(new_rets)
            print("Finish %d episode: avg. return: %.2f (+-%.2f) " % (len(returns),
                np.average(returns), np.std(returns) / np.sqrt(len(returns))))     
            
    savef.close()            

    if savevideo:
        gen_video(video_stats, os.path.join(saveimg_dir, name, "video.mp4"))    

if __name__ == "__main__":
    flags = util.parse(override=False)
    visualize(check_point_path=flags.load_checkpoint, 
              model_path=flags.preload_model, 
              visualize=False, 
              saveimg=True,
              savevideo=True)
