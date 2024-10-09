import torch
import torch.nn as nn
import numpy as np
import thinker
import thinker.util as util
import gym
import gym_sokoban
import pandas as pd
import numpy as np
from thinker.actor_net import DRCNet
from train_conv_probe import ConvProbe
import os
from thinker.actor_net import sample
from thinker.util import EnvOut
from typing import Optional
from run_agent_interv_exps import ProbeIntervDRCNet

num_trials = 200
fails = []
flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
for seed in [1,2]:
    fails = []
    dlocprobe2 = ConvProbe(32,5, 1, 0)
    dlocprobe2.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/250m_layer2_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe1 = ConvProbe(32,5, 1, 0)
    dlocprobe1.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/250m_layer1_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe0 = ConvProbe(32,5, 1, 0)
    dlocprobe0.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/250m_layer0_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    baseprobe0 = ConvProbe(32,5, 1, 0)
    baseprobe1 = ConvProbe(32,5, 1, 0)
    baseprobe2 = ConvProbe(32,5, 1, 0)
    box_probes = [dlocprobe0, dlocprobe1, dlocprobe2, baseprobe0, baseprobe1, baseprobe2]
    dlocprobe2 = ConvProbe(32,5, 1, 0)
    dlocprobe2.load_state_dict(torch.load(f"./convresults/models/agent_onto_after/250m_layer2_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe1 = ConvProbe(32,5, 1, 0)
    dlocprobe1.load_state_dict(torch.load(f"./convresults/models/agent_onto_after/250m_layer1_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe0 = ConvProbe(32,5, 1, 0)
    dlocprobe0.load_state_dict(torch.load(f"./convresults/models/agent_onto_after/250m_layer0_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    baseprobe0 = ConvProbe(32,5, 1, 0)
    baseprobe1 = ConvProbe(32,5, 1, 0)
    baseprobe2 = ConvProbe(32,5, 1, 0)
    agent_probes = [dlocprobe0, dlocprobe1, dlocprobe2, baseprobe0, baseprobe1, baseprobe2]
    #dloc_probes
    results = []
    for layer in [0,1,2,3,4,5]:
        for interv, (olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks) in exp_paths:
            for alpha in [0.5,1,2,4]:
                alpha_t = alpha
                if layer > 2:
                    alpha_a = alpha * agent_probes[layer%3].conv.weight.norm() / agent_probes[layer].conv.weight.norm()
                    alpha_b = alpha * box_probes[layer%3].conv.weight.norm() / box_probes[layer].conv.weight.norm()
                else:
                    alpha_a = alpha
                    alpha_b = alpha
                print(f"========================================= {layer=}, {alpha=}, {interv=}, {seed=}==================================")
                successes = 0
                alls = []
                for j in range(0,num_trials):
                    steps = []
                    env = thinker.make(
                                f"Sokoban-cutoffpusht4_clean_{j:04}-v0", 
                                env_n=1, 
                                gpu=False,
                                wrapper_type=1, 
                                has_model=False, 
                                train_model=False, 
                                parallel=False, 
                                save_flags=False,
                                mini=True,
                                mini_unqtar=False,
                                mini_unqbox=False         
                            ) 
                    if j == 0 and layer == 0 and seed == 0:
                        drc_net = DRCNet(
                                        obs_space=env.observation_space,
                                        action_space=env.action_space,
                                        flags=flags,
                                        record_state=True,
                                        )
                        ckp_path = "../drc_mini"
                        ckp_path = os.path.join(util.full_path(ckp_path), "ckp_actor_realstep250m.tar")
                        ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
                        drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
                        patch_net = ProbeIntervDRCNet(drc_net)

                    rnn_state = drc_net.initial_state(batch_size=1, device=env.device)
                    state = env.reset()
                    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

                    patch_old = True
                    fail = False
                    done = False
                    ep_len = 0
                    #levs = []

                    rot = j % 8
                    if rot in [3,5]:
                        right_idx = 1
                        left_idx = 2
                    elif rot in [1,7]:
                        right_idx = 2
                        left_idx = 1
                    elif rot in [2,4]:
                        right_idx = 3
                        left_idx = 4
                    elif rot in [0,6]:
                        right_idx = 4
                        left_idx = 3
                    else:
                        raise ValueError("index problem :(")

                    if rot in [0,4]:
                        down_idx = 2
                        up_idx = 1
                    elif rot in [1,5]:
                        down_idx = 3
                        up_idx = 4
                    elif rot in [2,6]:
                        down_idx = 1
                        up_idx = 2
                    elif rot in [3,7]:
                        down_idx = 4
                        up_idx = 3

                    while not done:
                        box_locs = (state["real_states"][0][2] == 1).to(int).view(-1).topk(k=(state["real_states"][0][2] == 1).to(int).sum()).indices.tolist()
                        notonstart = 0
                        for box_loc in box_locs:
                            box_x, box_y = box_loc % 8, (box_loc -(box_loc % 8))//8
                            if (box_y, box_x) in boxchecks[j]:
                                notonstart += 1 # need to fix this 
                                #print(box_y, box_x, checks[j], ep_len)
                        if notonstart != 1:
                            patch_old = False

                        if patch_old:
                            patch_info = {layer % 3: [{"vec": box_probes[layer].conv.weight[0].view(32), "locs": olds[j], "alpha": alpha_b},
                                        {"vec": box_probes[layer].conv.weight[right_idx].view(32), "locs": new_rs[j], "alpha": alpha_b},
                                        {"vec": box_probes[layer].conv.weight[left_idx].view(32), "locs": new_ls[j], "alpha": alpha_b},
                                        {"vec": box_probes[layer].conv.weight[down_idx].view(32), "locs": new_ds[j], "alpha": alpha_b},
                                        {"vec": box_probes[layer].conv.weight[up_idx].view(32), "locs": new_us[j], "alpha": alpha_b},
                                        {"vec": agent_probes[layer].conv.weight[up_idx].view(32), "locs": [(y+1, x) for (y, x) in new_rs[j]+new_ls[j]+new_ds[j]+new_us[j]], "alpha": alpha_a}] }
                        else:
                            patch_info = {layer % 3: [{"vec": box_probes[layer].conv.weight[0].view(32), "locs": olds[j], "alpha": 0}]}
                        patch_action, patch_action_probs, patch_logits, rnn_state, value = patch_net.forward_patch(env_out, rnn_state, activ_ticks=[0,1,2],
                                                                                patch_info=patch_info)
                        state, reward, done, info = env.step(patch_action)
                        ep_len += 1
                        env_out = util.create_env_out(patch_action, state, reward, done, info, flags)
                        #steps.append(state)
                    if ep_len < 115:
                        successes += 1
                        print(j)
                        #print(f"SUCCESS! {j=}, {fail=}, {ep_len}")
                    else:
                        pass
                    #else:
                        #print(f"{j=}, {fail=}, {ep_len}")
                        #fails.append(j)
                        #alls += [steps]
                print(successes / num_trials)
                results.append({"layer": layer, "alpha": alpha_t, "success_rate": (successes / num_trials), "intervs": interv})
    pd.DataFrame(results).to_csv(f"interv_results/cutoffinterv_seed{seed}.csv")