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

if __name__ == "__main__":
    paths_3intervs = [
        ([(3,1)], [(4,1),(4,2)], [], [], [(4,3)], [(3,1)], [(4,1)]),
        ([(2,1), (3,1)], [(4,1),(4,2)], [], [], [(4,3)], [(2,1)], [(4,1)]),
        ([(3,0), (4,0)], [(4,1), (4,2)], [], [], [(4,3)], [(3,0)], [(4,1)]),
        ([(4,0)], [(4,1)], [], [],[(4,2), (3,2)], [(4,0)], [(4,1)]),
        ([(2,3), (3,3)], [(4,3), (4,4), (4,5)], [], [], [], [(2,3)], [(4,3)]),
        ([(1,5), (2,5), (3,5)], [], [(4,5), (4,4)], [], [(4,3)], [(1,5)], [(4,5)]),
        ([(3,5)], [], [(4,5), (4,4), (4,3)], [], [], [(3,5)], [(4,5)]),
        ([(2,3)], [(3,3), (3,4), (3,5)], [], [], [], [(2,3)], [(3,3)]),
        ([(0,4)], [], [(2,4), (2,3)], [(1,4)], [], [(0,4)], [(1,4)]),
        ([(2,2)], [(3,2), (3,3), (3,4)], [], [], [], [(2,2)], [(3,2)]),
        ([(1,2), (2,2)], [(4,2), (4,3)], [], [(3,2)], [], [(1,2)], [(3,2)]),
        ([(3,4), (3,3)], [(1,2)], [], [], [(3,2), (2,2)], [(3,3)], [(3,2)]),
        ([(2,6)], [], [(1,6), (1,5)], [(1,4)], [], [(2,6)], [(1,6)]),
        ([(1,5)], [], [], [(1,6), (2,6), (3,6)], [], [(1,5)], [(1,6)]),
        ([(2,3), (2,4), (2,5)], [], [(4,6)], [(2,6), (3,6)], [], [(2,3)], [(2,6)]),
        ([(3,5)], [], [(4,5), (4,4)], [], [(4,3)], [(3,5)], [(4,5)]),
        ([(2,4)], [], [(1,4), (1,3), (1,2)], [], [], [(2,4)], [(1,4)]),
        ([(5,5), (4,5)], [], [(4,4), (4,3), (4,2)], [], [], [(5,5)], [(4,4)]),
        ([(1,2), (1,3)], [], [(3,4)], [(1,4), (2,4)], [], [(1,2)], [(1,4)]),
        ([(1,4), (1,5), (1,6)], [], [], [(2,6), (3,6), (4,6)], [], [(1,4)], [(2,6)])
    ]

    paths_2intervs = [
        ([(3,1)], [(4,1),(4,2)], [], [], [], [(3,1)], [(4,1)]),
        ([(2,1), (3,1)], [(4,1),(4,2)], [], [], [], [(2,1)], [(4,1)]),
        ([(3,0), (4,0)], [(4,1), (4,2)], [], [], [], [(3,0)], [(4,1)]),
        ([(4,0)], [(4,1)], [], [],[(4,2)], [(4,0)], [(4,1)]),
        ([(2,3), (3,3)], [(4,3), (4,4)], [], [], [], [(2,3)], [(4,3)]),
        ([(1,5), (2,5), (3,5)], [], [(4,5), (4,4)], [], [], [(1,5)], [(4,5)]),
        ([(3,5)], [], [(4,5), (4,4)], [], [], [(3,5)], [(4,5)]),
        ([(2,3)], [(3,3), (3,4)], [], [], [], [(2,3)], [(3,3)]),
        ([(0,4)], [], [(2,4)], [(1,4)], [], [(0,4)], [(1,4)]),
        ([(2,2)], [(3,2), (3,3)], [], [], [], [(2,2)], [(3,2)]),
        ([(1,2), (2,2)], [(4,2)], [], [(3,2)], [], [(1,2)], [(3,2)]),
        ([(3,4), (3,3)], [], [], [], [(3,2), (2,2)], [(3,3)], [(3,2)]),
        ([(2,6)], [], [(1,6), (1,5)], [], [], [(2,6)], [(1,6)]),
        ([(1,5)], [], [], [(1,6), (2,6)], [], [(1,5)], [(1,6)]),
        ([(2,3), (2,4), (2,5)], [], [], [(2,6), (3,6)], [], [(2,3)], [(2,6)]),
        ([(3,5)], [], [(4,5), (4,4)], [], [], [(3,5)], [(4,5)]),
        ([(2,4)], [], [(1,4), (1,3)], [], [], [(2,4)], [(1,4)]),
        ([(5,5), (4,5)], [], [(4,4), (4,3)], [], [], [(5,5)], [(4,4)]),
        ([(1,2), (1,3)], [], [], [(1,4), (2,4)], [], [(1,2)], [(1,4)]),
        ([(1,4), (1,5), (1,6)], [], [], [(2,6), (3,6)], [], [(1,4)], [(2,6)])
    ]

    paths_1intervs = [
        ([(3,1)], [(4,1)], [], [], [], [(3,1)], [(4,1)]),
        ([(2,1), (3,1)], [(4,1)], [], [], [], [(2,1)], [(4,1)]),
        ([(3,0), (4,0)], [(4,1)], [], [], [], [(3,0)], [(4,1)]),
        ([(4,0)], [(4,1)], [], [],[], [(4,0)], [(4,1)]),
        ([(2,3), (3,3)], [(4,3)], [], [], [], [(2,3)], [(4,3)]),
        ([(1,5), (2,5), (3,5)], [], [(4,5)], [], [], [(1,5)], [(4,5)]),
        ([(3,5)], [], [(4,5)], [], [], [(3,5)], [(4,5)]),
        ([(2,3)], [(3,3)], [], [], [], [(2,3)], [(3,3)]),
        ([(0,4)], [], [], [(1,4)], [], [(0,4)], [(1,4)]),
        ([(2,2)], [(3,2)], [], [], [], [(2,2)], [(3,2)]),
        ([(1,2), (2,2)], [], [], [(3,2)], [], [(1,2)], [(3,2)]),
        ([(3,4), (3,3)], [], [], [], [(3,2)], [(3,3)], [(3,2)]),
        ([(2,6)], [], [(1,6)], [], [], [(2,6)], [(1,6)]),
        ([(1,5)], [], [], [(1,6)], [], [(1,5)], [(1,6)]),
        ([(2,3), (2,4), (2,5)], [], [], [(2,6)], [], [(2,3)], [(2,6)]),
        ([(3,5)], [], [(4,5)], [], [(4,3)], [(3,5)], [(4,5)]),
        ([(2,4)], [], [(1,4)], [], [], [(2,4)], [(1,4)]),
        ([(5,5), (4,5)], [], [(4,4)], [], [], [(5,5)], [(4,4)]),
        ([(1,2), (1,3)], [], [], [(1,4)], [], [(1,2)], [(1,4)]),
        ([(1,4), (1,5), (1,6)], [], [], [(2,6)], [], [(1,4)], [(2,6)])
    ]

    paths_0intervs = [
        ([(3,1)], [], [], [], [], [(3,1)], [(4,1)]),
        ([(2,1), (3,1)], [], [], [], [], [(2,1)], [(4,1)]),
        ([(3,0), (4,0)], [], [], [], [], [(3,0)], [(4,1)]),
        ([(4,0)], [], [], [],[], [(4,0)], [(4,1)]),
        ([(2,3), (3,3)], [], [], [], [], [(2,3)], [(4,3)]),
        ([(1,5), (2,5), (3,5)], [], [], [], [], [(1,5)], [(4,5)]),
        ([(3,5)], [], [], [], [], [(3,5)], [(4,5)]),
        ([(2,3)], [], [], [], [], [(2,3)], [(3,3)]),
        ([(0,4)], [], [], [], [], [(0,4)], [(1,4)]),
        ([(2,2)], [], [], [], [], [(2,2)], [(3,2)]),
        ([(1,2), (2,2)], [], [], [], [], [(1,2)], [(3,2)]),
        ([(3,4), (3,3)], [], [], [], [], [(3,3)], [(3,2)]),
        ([(2,6)], [], [], [], [], [(2,6)], [(1,6)]),
        ([(1,5)], [], [], [], [], [(1,5)], [(1,6)]),
        ([(2,3), (2,4), (2,5)], [], [], [], [], [(2,3)], [(2,6)]),
        ([(3,5)], [], [], [], [(4,3)], [(3,5)], [(4,5)]),
        ([(2,4)], [], [], [], [], [(2,4)], [(1,4)]),
        ([(5,5), (4,5)], [], [], [], [], [(5,5)], [(4,4)]),
        ([(1,2), (1,3)], [], [], [], [], [(1,2)], [(1,4)]),
        ([(1,4), (1,5), (1,6)], [], [], [], [], [(1,4)], [(2,6)])
    ]
    exp_paths = []
    for name, paths in zip([0, 1, 2, 3], [paths_0intervs, paths_1intervs, paths_2intervs, paths_3intervs]):
        olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks = [], [], [], [], [], [], []
        for old_path, new_right, new_left, new_down, new_up, checkpoint, boxpoint in paths:
            old_path_1 = [(x,7-y) for y,x in old_path]
            new_right_1 = [(x,7-y) for y,x in new_right]
            new_left_1 = [(x,7-y) for y,x in new_left]
            new_down_1 = [(x,7-y) for y,x in new_down]
            new_up_1 = [(x,7-y) for y,x in new_up]
            checkpoint_1 = [(x,7-y) for y,x in checkpoint]
            boxpoint_1 = [(x,7-y) for y,x in boxpoint]

            old_path_2 = [(7-y,7-x) for y,x in old_path]
            new_right_2 = [(7-y,7-x) for y,x in new_right]
            new_left_2 = [(7-y,7-x) for y,x in new_left]
            new_down_2 = [(7-y,7-x) for y,x in new_down]
            new_up_2 = [(7-y,7-x) for y,x in new_up]
            checkpoint_2 = [(7-y,7-x) for y,x in checkpoint]
            boxpoint_2 = [(7-y,7-x) for y,x in boxpoint]

            old_path_3 = [(7-x,y) for y,x in old_path]
            new_right_3 = [(7-x,y) for y,x in new_right]
            new_left_3 = [(7-x,y) for y,x in new_left]
            new_down_3 = [(7-x,y) for y,x in new_down]
            new_up_3 = [(7-x,y) for y,x in new_up]
            checkpoint_3 = [(7-x,y) for y,x in checkpoint]
            boxpoint_3 = [(7-x,y) for y,x in boxpoint]

            old_path_4 = [(y,7-x) for y,x in old_path]
            new_right_4 = [(y,7-x) for y,x in new_right]
            new_left_4 = [(y,7-x) for y,x in new_left]
            new_down_4 = [(y,7-x) for y,x in new_down]
            new_up_4 = [(y,7-x) for y,x in new_up]
            checkpoint_4 = [(y,7-x) for y,x in checkpoint]
            boxpoint_4 = [(y,7-x) for y,x in boxpoint]

            old_path_5 = [(x,7-y) for y,x in old_path_4]
            new_right_5 = [(x,7-y) for y,x in new_right_4]
            new_left_5 = [(x,7-y) for y,x in new_left_4]
            new_down_5 = [(x,7-y) for y,x in new_down_4]
            new_up_5 = [(x,7-y) for y,x in new_up_4]
            checkpoint_5 = [(x,7-y) for y,x in checkpoint_4]
            boxpoint_5 = [(x,7-y) for y,x in boxpoint_4]

            old_path_6 = [(7-y,7-x) for y,x in old_path_4]
            new_right_6 = [(7-y,7-x) for y,x in new_right_4]
            new_left_6 = [(7-y,7-x) for y,x in new_left_4]
            new_down_6 = [(7-y,7-x) for y,x in new_down_4]
            new_up_6 = [(7-y,7-x) for y,x in new_up_4]
            checkpoint_6 = [(7-y,7-x)  for y,x in checkpoint_4]
            boxpoint_6 = [(7-y,7-x)  for y,x in boxpoint_4]

            old_path_7 = [(7-x,y) for y,x in old_path_4]
            new_right_7 = [(7-x,y) for y,x in new_right_4]
            new_left_7 = [(7-x,y) for y,x in new_left_4]
            new_down_7 = [(7-x,y) for y,x in new_down_4]
            new_up_7 = [(7-x,y) for y,x in new_up_4]
            checkpoint_7 = [(7-x,y) for y,x in checkpoint_4]
            boxpoint_7 = [(7-x,y) for y,x in boxpoint_4]

            olds += [old_path, old_path_1, old_path_2, old_path_3, old_path_4, old_path_5, old_path_6, old_path_7]
            new_rs += [new_right, new_right_1, new_right_2, new_right_3, new_right_4, new_right_5, new_right_6, new_right_7]
            new_ls += [new_left, new_left_1, new_left_2, new_left_3, new_left_4, new_left_5, new_left_6, new_left_7]
            new_ds += [new_down, new_down_1, new_down_2, new_down_3, new_down_4, new_down_5, new_down_6, new_down_7]
            new_us += [new_up, new_up_1, new_up_2, new_up_3, new_up_4, new_up_5, new_up_6, new_up_7]
            checks += [checkpoint, checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4, checkpoint_5, checkpoint_6, checkpoint_7]
            boxchecks += [boxpoint, boxpoint_1, boxpoint_2, boxpoint_3, boxpoint_4, boxpoint_5, boxpoint_6, boxpoint_7]
        exp_paths.append([name, (olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks)])

    num_trials = 160
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    for seed in [0,1,2,3,4]:
        fails = []
        dlocprobe2 = ConvProbe(32,5, 1, 0)
        dlocprobe2.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/250m_layer2_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
        dlocprobe1 = ConvProbe(32,5, 1, 0)
        dlocprobe1.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/250m_layer1_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
        dlocprobe0 = ConvProbe(32,5, 1, 0)
        dlocprobe0.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/250m_layer0_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
        dloc_probes = [dlocprobe0, dlocprobe1, dlocprobe2]
        results = []
        for layer in [0,1,2]:
            for interv, (olds, new_rs, new_ls, new_ds, new_us, checks, boxchecks) in exp_paths:
                for alpha in [0.25,0.5,1,2,4]:
                    successes = 0
                    for j in range(num_trials):
                        env = thinker.make(
                                    f"Sokoban-boxshortcut_clean_{j:04}-v0", 
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
                        levs = []

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
                                if (box_y, box_x) in checks[j]:
                                    fail = True
                            if notonstart != 1:
                                patch_old = False

                            if patch_old:
                                patch_info = {layer: [{"vec": dloc_probes[layer].conv.weight[0].view(32), "locs": olds[j], "alpha": alpha},
                                            {"vec": dloc_probes[layer].conv.weight[right_idx].view(32), "locs": new_rs[j], "alpha": alpha},
                                            {"vec": dloc_probes[layer].conv.weight[left_idx].view(32), "locs": new_ls[j], "alpha": alpha},
                                            {"vec": dloc_probes[layer].conv.weight[down_idx].view(32), "locs": new_ds[j], "alpha": alpha},
                                            {"vec": dloc_probes[layer].conv.weight[up_idx].view(32), "locs": new_us[j], "alpha": alpha}] }
                            else:
                                patch_info = {layer: [{"vec": dloc_probes[layer].conv.weight[0].view(32), "locs": olds[j], "alpha": alpha}]}
                            patch_action, patch_action_probs, patch_logits, rnn_state, value = patch_net.forward_patch(env_out, rnn_state, activ_ticks=[0,1,2],
                                                                                    patch_info=patch_info)
                            state, reward, done, info = env.step(patch_action)
                            ep_len += 1
                            env_out = util.create_env_out(patch_action, state, reward, done, info, flags)
                            levs.append(state["real_states"][0])
                        if not fail and ep_len < 115:
                            successes += 1
                    print(successes / num_trials)
                    results.append({"layer": layer, "alpha": alpha, "success_rate": (successes / num_trials), "intervs": interv})
        pd.DataFrame(results).to_csv(f"interv_results/boxinterv_seed{seed}.csv")
