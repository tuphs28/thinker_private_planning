import thinker
import thinker.util as util
import os
import torch
from thinker.actor_net import DRCNet
import pandas as pd
from actpatchdrc import ActPatchDRCNet
from typing import Optional
import numpy as np
import gym
import gym_sokoban

#exp = "cutoffpush"

def run_act_patch_exps(exp: str, layer_channel_dict_list: list, mode: str, inter_ticks: list, final_ticks: list, num_steps: int, eval_metric: str = "prob", gpu: bool = False, disp_level_ids: bool = False):

    mini_sokoban = True
    mini_unqtar = False
    mini_unqbox = False
    env_n = 1
    all_results = []
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    flags.mini = mini_sokoban
    flags.mini_unqbtar = mini_unqtar
    flags.mini_unqbox = mini_unqbox

    num_levels = len(os.listdir(f"../sokoban/gym_sokoban/envs/boxoban-levels/experiments/{exp}"))

    for i in range(num_levels):
        if disp_level_ids:
            print(f"------------- Level {i:04} -----------")
        results = {}
        env = thinker.make(
            f"Sokoban-{exp}_corrupt_{i:04}-v0", 
            env_n=env_n, 
            gpu=gpu,
            wrapper_type=1, 
            has_model=False, 
            train_model=False, 
            parallel=False, 
            save_flags=False,
            mini=mini_sokoban,
            mini_unqtar=mini_unqtar,
            mini_unqbox=mini_unqbox         
        ) 

        if i == 0:
            drc_net = DRCNet(
                obs_space=env.observation_space,
                action_space=env.action_space,
                flags=flags,
                record_state=True,
                )
            drc_net.to(env.device)
            ckp_path = "../drc_mini"
            ckp_path = os.path.join(util.full_path(ckp_path), "ckp_actor_realstep249000192.tar")
            ckp = torch.load(ckp_path, env.device)
            drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
            patch_net = ActPatchDRCNet(drc_net)

        clean_activs = []
        corrupt_activs = []
        clean_actions = []

        rnn_state = drc_net.initial_state(batch_size=env_n, device=env.device)
        state = env.reset()
        env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
        for step in range(num_steps):
            actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
            clean_activs.append(drc_net.hidden_state[:,1:,:,:])
            state, reward, done, info = env.step(actor_out.action)
            clean_actions.append(actor_out.action)
            env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)

        clean_loc = state["real_states"][0,[4,5],:,:].sum(dim=0).argmax()
        env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
        actor_out, _ = drc_net(env_out, rnn_state, greedy=True)
        clean_activs.append(drc_net.hidden_state[:,1:,:,:])

        clean_probs = actor_out.action_prob.view(-1)
        clean_logits = actor_out.pri_param.view(-1).detach()
        clean_action_idx = actor_out.action_prob.view(-1).argmax().item()
        #print("clean:", actor_out.action_prob.view(-1).tolist())

        env = thinker.make(
            f"Sokoban-{exp}_clean_{i:04}-v0", 
            env_n=env_n, 
            gpu=gpu,
            wrapper_type=1, 
            has_model=False, 
            train_model=False, 
            parallel=False, 
            save_flags=False,
            mini=mini_sokoban,
            mini_unqtar=mini_unqtar,
            mini_unqbox=mini_unqbox         
            ) 

        rnn_state = drc_net.initial_state(batch_size=env_n, device=env.device)
        state = env.reset()
        env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
        for step in range(num_steps):
            actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
            corrupt_activs.append(drc_net.hidden_state[:,1:,:,:])
            state, reward, done, info = env.step(actor_out.action)
            env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
 
        corrupt_loc = state["real_states"][0,[4,5],:,:].sum(dim=0).argmax()
        env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
        actor_out, _ = drc_net(env_out, rnn_state, greedy=True)
        corrupt_activs.append(drc_net.hidden_state[:,1:,:,:])

        corrupt_probs = actor_out.action_prob.view(-1)
        corrupt_logits = actor_out.pri_param.view(-1).detach()
        corrupt_action_idx = actor_out.action_prob.view(-1).argmax().item()
        #print("corrupt:", actor_out.action_prob.view(-1).tolist())

        assert clean_loc == corrupt_loc, f"Agent is not at the same location at timestep {num_steps} for environment {i:04}: {clean_loc=},{corrupt_loc=}"

        if eval_metric == "ld":
            clean_ld = (clean_logits[clean_action_idx] - clean_logits[corrupt_action_idx]).item()
            corrupt_ld = (corrupt_logits[clean_action_idx] - corrupt_logits[corrupt_action_idx]).item()
            diff_ld = clean_ld - corrupt_ld
        elif eval_metric == "prob":
            corrupt_action_prob = corrupt_probs[clean_action_idx].item()

        for patch_dict in layer_channel_dict_list:

            rnn_state = drc_net.initial_state(batch_size=env_n, device=env.device)
            state = env.reset() 
            env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
            for step in range(num_steps):
                patch_action, patch_action_probs, patch_logits, rnn_state = patch_net.forward_patch(env_out, rnn_state, activ_type=mode, activ_ticks=inter_ticks,
                                                                    patch_dict=patch_dict, activs=clean_activs[step])
                state, reward, done, info = env.step(patch_action)
                env_out = util.create_env_out(patch_action, state, reward, done, info, flags)

            patch_loc = state["real_states"][0,[4,5],:,:].sum(dim=0).argmax()
            assert patch_loc == corrupt_loc, f"In patched episode with patched channels {patch_dict} agent is not at the same location as clean and currupt episodes at timestep {num_steps} for environment {i:04}: {patch_loc=},{corrupt_loc=}"

            patch_action, patch_action_probs, patch_logits, _ = patch_net.forward_patch(env_out, rnn_state, activ_type=mode, activ_ticks=final_ticks,
                                                                patch_dict=patch_dict, activs=clean_activs[num_steps])
            
            if eval_metric == "ld":
                patch_ld = (patch_logits[clean_action_idx] - patch_logits[corrupt_action_idx]).item()
                patch_metric = (patch_ld - corrupt_ld) / diff_ld
            elif eval_metric == "prob": 
                patch_metric = patch_action_probs[clean_action_idx].item() - corrupt_action_prob

            #print("patch:", patch_action_probs.view(-1).tolist(), "metric: ", patch_metric)
            results[f"{patch_dict}"] = round(patch_metric,4)

        all_results.append(results)
    return all_results

def run_act_patch_topk(exp: str, k: int, layer_channel_dict: dict, mode: str, inter_ticks: list, final_ticks: list, num_steps: int, eval_metric: str = "prob", gpu: bool = False, disp_level_ids: bool = False):
    all_results = {}
    exp_results = run_act_patch_exps(exp=exp, layer_channel_dict_list=[layer_channel_dict], mode=mode, inter_ticks=inter_ticks, final_ticks=final_ticks, num_steps=num_steps, gpu=gpu, disp_level_ids=disp_level_ids)
    all_results[f"{layer_channel_dict}"] = list(exp_results[0].values())[0]

    channels = []
    for layer, layer_channels in layer_channel_dict.items():
        channels += [layer*32+c for c in layer_channels]
    all_channel_subsets = [channels[:i]+channels[i+1:] for i in range(len(channels))]
    layer_channel_dict_list = []
    for channels_subset in all_channel_subsets:
        layer_channel_dict_list.append({
            0: [c%32 for c in channels_subset if c<32],
            1: [c%32 for c in channels_subset if c>=32 and c<64],
            2: [c%32 for c in channels_subset if c>=64]
        })
    
    while len(all_channel_subsets) > k:
        exp_results = run_act_patch_exps(exp=exp, layer_channel_dict_list=layer_channel_dict_list, mode=mode, inter_ticks=inter_ticks, final_ticks=final_ticks, num_steps=num_steps, gpu=gpu, disp_level_ids=disp_level_ids)
        patch_effects = np.array([list(exp_result.values()) for exp_result in exp_results]).mean(axis=0)
        top_subset_dict = layer_channel_dict_list[patch_effects.argmax()]
        print(top_subset_dict)
        print(patch_effects.max())
        all_results[f"{top_subset_dict}"] = patch_effects.max()

        channels = []
        for layer, layer_channels in top_subset_dict.items():
            channels += [layer*32+c for c in layer_channels]
        all_channel_subsets = [channels[:i]+channels[i+1:] for i in range(len(channels))]
        layer_channel_dict_list = []
        for channels_subset in all_channel_subsets:
            layer_channel_dict_list.append({
                0: [c%32 for c in channels_subset if c<32],
                1: [c%32 for c in channels_subset if c>=32 and c<64],
                2: [c%32 for c in channels_subset if c>=64]
            })

    return top_subset_dict, all_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="run activation patching exps")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--expmode", type=str, default="topkprune")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--numsteps", type=int, required=True)
    parser.add_argument("--allsteps", type=bool, default=True)
    parser.add_argument("--gpu", type=bool, default=False)
    args = parser.parse_args()

    mode = "cell"
    inter_ticks = [0,1,2] if args.allsteps else []
    exp_name = f"{args.exp}_{args.expmode}__k{args.k}_layer{args.layer}_{'allsteps' if args.allsteps else 'finalstep'}"
    print(inter_ticks)
    final_ticks = [0]
    print(args.k, type(args.k))

    if args.expmode == "allk":
        if args.k == 1:
            layer_channel_dict_list= [
                {args.layer: [a]} for a in range(32)
            ]
        elif args.k == 2:
            layer_channel_dict_list= [
                {args.layer: [a,b]} for a in range(32) for b in range(32) if (b>a)
            ]
        elif args.k == 3:
            layer_channel_dict_list= [
                {args.layer: [a,b,c]} for a in range(32) for b in range(32) for c in range(32) if (c>b and b>a)
            ]
        else:
            raise ValueError("k>3 is not currently supported for allk")
        exp_results = run_act_patch_exps(exp=args.exp, layer_channel_dict_list=layer_channel_dict_list, mode=mode, inter_ticks=inter_ticks, final_ticks=final_ticks, num_steps=args.numsteps, gpu=args.gpu, disp_level_ids=True)
        results_df = pd.DataFrame(exp_results).T
    elif args.expmode == "topkprune":
        layer_channel_dict = {2: [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,25,26,27,28,29,30,31]}
        top_subset, exp_results = run_act_patch_topk(exp=args.exp, k=args.k, layer_channel_dict=layer_channel_dict, mode=mode, inter_ticks=inter_ticks, final_ticks=final_ticks, num_steps=args.numsteps, gpu=args.gpu, disp_level_ids=True)
        results_df = pd.DataFrame(exp_results.values(), index=[f"{subset_dict}" for subset_dict in exp_results.keys()])

    results_df.to_csv((f"./results/{exp_name}_{mode}_TEST7.csv"))