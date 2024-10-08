import torch
from create_probe_dataset import ProbingDataset, make_current_board_feature_detector, make_tar_info_extractor, make_agent_info_extractor, make_box_info_extractor
import numpy as np
import thinker
import thinker.viz_utils as viz
import thinker.util as util
import gym
import gym_sokoban
import pandas as pd
import numpy as np
from thinker.actor_net import DRCNet
from train_conv_probe import ConvProbe
from sklearn.metrics import precision_recall_fscore_support
import os

env_name = ""
checkpoint = 20
gpu = False
num_episodes = 100
debug = False
seed = 0
num_steps = 6
thinking_steps = 6

results = {}
for checkpoint in range(1,51):
    print(f"========== {checkpoint} =============")
    env = thinker.make(
        f"Sokoban-{env_name}v0", 
        env_n=1, 
        gpu=gpu,
        wrapper_type=1, 
        has_model=False, 
        train_model=False, 
        parallel=False, 
        save_flags=False,
        mini=True,
        mini_unqtar=False,
        mini_unqbox=False         
        ) 
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    flags.mini = True
    flags.mini_unqtar = False
    flags.mini_unqbox = False
    drc_net = DRCNet(
        obs_space=env.observation_space,
        action_space=env.action_space,
        flags=flags,
        record_state=True,
    )
    ckp_path = "../drc_mini"
    ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{checkpoint}m.tar")

    adj_wall_detector = make_current_board_feature_detector(feature_idxs=[0], mode="adj")
    adj_boxnotontar_detector = make_current_board_feature_detector(feature_idxs=[2], mode="adj")
    adj_boxontar_detector = make_current_board_feature_detector(feature_idxs=[3], mode="adj")
    adj_box_detector = make_current_board_feature_detector(feature_idxs=[2,3], mode="adj")
    adj_tar_detector = make_current_board_feature_detector(feature_idxs=[6], mode="adj")
    num_boxnotontar_detector = make_current_board_feature_detector(feature_idxs=[2], mode="num")
    agent_loc_detector = make_current_board_feature_detector(feature_idxs=[4,5], mode="loc")
    box_loc_detector = make_current_board_feature_detector(feature_idxs=[2,3], mode="loc")
    tar_loc_detector = make_current_board_feature_detector(feature_idxs=[3,5,6], mode="loc")
    boxontar_loc_detector = make_current_board_feature_detector(feature_idxs=[3], mode="loc")
    justtar_loc_detector = make_current_board_feature_detector(feature_idxs=[5,6], mode="loc")
    boxnotontar_loc_detector = make_current_board_feature_detector(feature_idxs=[2], mode="loc")
    noboxdetector =  make_current_board_feature_detector(feature_idxs=[1,6], mode="loc")

    ckp = torch.load(ckp_path, env.device)
    drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    drc_net.to(env.device)

    current_board_feature_fncs = [
        ("adj_walls", adj_wall_detector),
        ("adj_boxnotontar", adj_boxnotontar_detector),
        ("adj_boxontar", adj_boxontar_detector),
        ("adj_box", adj_box_detector),
        ("adj_tar", adj_tar_detector),
        ("num_boxnotontar", num_boxnotontar_detector),
        ("agent_loc", agent_loc_detector),
        ("box_loc", box_loc_detector),
        ("tar_loc", tar_loc_detector),
        ("boxontar_loc", boxontar_loc_detector),
        ("boxnotontar_loc", boxnotontar_loc_detector),
        ("justtar_loc", justtar_loc_detector),
        ("nobox_loc", noboxdetector)
    ]
    future_feature_fncs = [make_tar_info_extractor(unq=False), make_box_info_extractor(unq=False), make_agent_info_extractor()]

    dlocprobe2 = ConvProbe(32,5, 1, 0)
    dlocprobe2.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/{checkpoint}m_layer2_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe1 = ConvProbe(32,5, 1, 0)
    dlocprobe1.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/{checkpoint}m_layer1_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe0 = ConvProbe(32,5, 1, 0)
    dlocprobe0.load_state_dict(torch.load(f"./convresults/models/tracked_box_next_push_onto_with/{checkpoint}m_layer0_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    box_probes = [dlocprobe0, dlocprobe1, dlocprobe2]

    dlocprobe2 = ConvProbe(32,5, 1, 0)
    dlocprobe2.load_state_dict(torch.load(f"./convresults/models/agent_onto_after/{checkpoint}m_layer2_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe1 = ConvProbe(32,5, 1, 0)
    dlocprobe1.load_state_dict(torch.load(f"./convresults/models/agent_onto_after/{checkpoint}m_layer1_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    dlocprobe0 = ConvProbe(32,5, 1, 0)
    dlocprobe0.load_state_dict(torch.load(f"./convresults/models/agent_onto_after/{checkpoint}m_layer0_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
    agent_probes = [dlocprobe0, dlocprobe1, dlocprobe2]

    rnn_state = drc_net.initial_state(batch_size=1, device=env.device)
    state = env.reset() 
    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

    episode_length = 0
    board_num = 0
    probing_data = []
    episode_entry = []

    actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
    trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
    trans_entry["action"] = actor_out.action.item()
    trans_entry["value"] = round(actor_out.baseline.item(), 3) 
    trans_entry["board_state"] = state["real_states"][0].detach().cpu() # tensor of size (channels, board_height, board_width)
    trans_entry["hidden_states"] = drc_net.hidden_state[0].detach().cpu() # tensor of size (ticks+1, layers*64, representation_height, representation_width)
    trans_entry["board_num"] = board_num
    episode_length += 1

    while(board_num < num_episodes):

        state, reward, done, info = env.step(actor_out.action if episode_length >= thinking_steps else torch.tensor([0]))
        trans_entry["reward"] = round(reward.item(), 3) # round rewards to 3 d.p.

        
        if episode_length < num_steps:
            for tick in [1,2,3]:
                for layer, boxprobe in enumerate(box_probes):
                    logits, _ = boxprobe(drc_net.core.hidden_state[0,tick,(layer*64)+32:(layer*64)+64,:,:])
                    trans_entry[f"plan_box_layer{layer+1}_tick_{tick}"] = logits.argmax(dim=0)
                for layer, agentprobe in enumerate(agent_probes):
                    logits, _ = agentprobe(drc_net.core.hidden_state[0,tick,(layer*64)+32:(layer*64)+64,:,:])
                    trans_entry[f"plan_agent_layer{layer+1}_tick_{tick}"] = logits.argmax(dim=0)
        episode_entry.append(trans_entry)

        if done:
            for fnc in future_feature_fncs:
                episode_entry = fnc(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry["steps_remaining"] = episode_length - trans_idx
                trans_entry["steps_taken"] = trans_idx+1
                trans_entry["return"] = sum([(0.97**t)*future_trans["reward"] for t, future_trans in enumerate(episode_entry[trans_idx:])])  
            
            if episode_length < 140: 
                probing_data += episode_entry 
            
            episode_length = 0
            board_num += 1
            print("Data collected from episode", board_num, "with episode length of", len(episode_entry))
            episode_entry = []
            rnn_state = drc_net.initial_state(batch_size=1, device=env.device)

        env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
        actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
        if debug:
            print(actor_out.pri_param.argmax(dim=-1).item(), actor_out.action.item())

        trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
        trans_entry["action"] = actor_out.action.item()
        trans_entry["value"] = round(actor_out.baseline.item(), 3) 
        trans_entry["board_state"] = state["real_states"][0].detach().cpu() # tensor of size (channels, board_height, board_width)
        trans_entry["hidden_states"] = drc_net.hidden_state[0].detach().cpu() # tensor of size (ticks+1, layers*64, representation_height, representation_width)
        trans_entry["board_num"] = board_num

        episode_length += 1

    checkpoint_results = {}
    for layer in [0,1,2]:
        labs_a, preds_a = [[] for i in range((num_steps-1)*3)], [[] for i in range((num_steps-1)*3)]
        for trans in probing_data:
            if trans["steps_taken"] < num_steps:
                for tick in range(3):
                    labs_a[(trans["steps_taken"]-1)*3 + tick] += trans["agent_onto_after"].view(-1).tolist()
                    preds_a[(trans["steps_taken"]-1)*3 + tick] += trans[f"plan_agent_layer{layer+1}_tick_{tick+1}"].view(-1).tolist()
        for i in range(len(labs_a)):
            prec, rec, f1, sup = precision_recall_fscore_support(labs_a[i], preds_a[i], average='macro', zero_division=1)
            checkpoint_results[f"plan_agent_layer{layer+1}_tick_{tick+1}"] = f1.item()

        labs_b, preds_b = [[] for i in range((num_steps-1)*3)], [[] for i in range((num_steps-1)*3)]
        for trans in probing_data:
            if trans["steps_taken"] < num_steps:
                for tick in range(3):
                    labs_b[(trans["steps_taken"]-1)*3 + tick] += trans["tracked_box_next_push_onto_with"].view(-1).tolist()
                    preds_b[(trans["steps_taken"]-1)*3 + tick] += trans[f"plan_box_layer{layer+1}_tick_{tick+1}"].view(-1).tolist()
        for i in range(len(labs_b)):
            prec, rec, f1, sup = precision_recall_fscore_support(labs_b[i], preds_b[i], average='macro', zero_division=1)
            checkpoint_results[f"plan_box_layer{layer+1}_tick_{tick+1}"] = f1.item()

    results[checkpoint] = checkpoint_results
    pd.DataFrame(results).to_csv("./planaccs_over_ticks.csv")