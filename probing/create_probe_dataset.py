from thinker.main import make, Env
import thinker
from thinker.actor_net import DRCNet
from torch.utils.data.dataset import Dataset
import torch
from torch.nn.functional import relu
from thinker import util
from typing import Callable, NamedTuple, Optional
from numpy.random import uniform
import os
import argparse

def make_current_board_feature_detector(feature_idxs: list, mode: str) -> Callable:
    """Create feature detector functions to extract discrete features from mini-sokoban boards. Boards must be (7,8,8) arrays

    Args:
        feature_idxs (list): list index of feature of interest (see sokoban.cpp);
        mode (str): type of feature detector to construct: "adj" (to count number of adjacent features), "num" (to count total number of features on board) or "loc" (extract location of features)

    Returns:
        Callable: feature detector function, takes in a board state and returns the desired feature
    """
    if mode == "adj":
        def feature_detector(board: torch.tensor) -> int:
            h, w = board.shape[1:]
            x, y = ((board[4,:,:]==1) + (board[5,:,:]==1)).nonzero()[0,:]
            adj_coords = [(xp, yp) for xp, yp in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if xp>-1 and xp<h and yp>-1 and yp<w]
            n_hits = 0
            for (xp,yp) in adj_coords:
                for feature_idx in feature_idxs:
                    if board[feature_idx, xp, yp] == 1:
                        n_hits += 1
            return n_hits
    elif mode == "num":
        def feature_detector(board: torch.tensor) -> int:
            return sum([torch.sum((board[feature_idx,:,:]==1).int()) for feature_idx in feature_idxs]).item()
    elif mode == "loc":
        def feature_detector(board):
            locs_xy = sum([(board[feature_idx,:,:]==1) for feature_idx in feature_idxs]).nonzero()
            locs = tuple([(8*x+y).item() for (x,y) in locs_xy]) # each location is an int in range [0,63]
            return locs
    else:
        raise ValueError(f"Please enter a valid mode to construct a feature detector - user entered {mode}, valid modes are adj, num and loc")
    return feature_detector

def make_future_feature_detector(feature_name: str, mode: str, steps_ahead: Optional[int] = None) -> Callable:
    """Create function that adds a feature to each transition (i.e. a dictionary of features) corresponding to the feature with name feature_name in steps_ahead steps

    Args:
        feature_name (str): feature to track steps_ahead into the future
        steps_ahead (Optional int): number of steps ahead into the future to look for this feature if mode is either ahead or traj
        mode (str): type of feature detector to construct: ahead (make feature corresponding to feature_name in steps_ahead steps), traj (make feature corresponding to trajectory of feature_name from current value to over steps_ahead steps) or change (number of steps until the feature next changes)

    Returns:
        Callable: feature detector function, takes in a list of transitions for a single episode and adds an entry for feature_name in steps_ahead steps
    """
    if mode == "ahead":
        new_feature_name = f"{feature_name}_ahead_{steps_ahead}"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry[new_feature_name] = episode_entry[trans_idx+steps_ahead][feature_name] if trans_idx < episode_length-steps_ahead-1 else -1
            return episode_entry
    elif mode == "traj":
        new_feature_name = f"{feature_name}_traj_{steps_ahead}"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                traj = []
                if trans_idx < episode_length-steps_ahead-1:
                    for traj_idx in range(steps_ahead+1):
                        traj.append(episode_entry[trans_idx+traj_idx][feature_name])
                    trans_entry[new_feature_name] = tuple(traj)
                else:
                    trans_entry[new_feature_name] = -1
            return episode_entry
    elif mode == "change":
        new_feature_name = f"{feature_name}_until_change"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            if type(episode_entry[0][feature_name]) == int or type(episode_entry[0][feature_name]) == float:
                for trans_idx, trans_entry in enumerate(episode_entry):
                    future_idx = 0
                    while episode_entry[trans_idx+future_idx][feature_name] == trans_entry[feature_name]:
                        future_idx += 1
                        if trans_idx + future_idx == episode_length: # if no change in feature over rest of episode, just count steps until end of episode (e.g. this is the desired behaviour for counting boxes) - may need to change this if using for features over than boxes
                            break
                    trans_entry[new_feature_name] = future_idx
            elif type(episode_entry[0][feature_name]) == tuple:
                for trans_idx, trans_entry in enumerate(episode_entry):
                    future_idx = 0
                    current_tensor = torch.tensor(episode_entry[trans_idx][feature_name])
                    while trans_idx + future_idx < episode_length - 2:
                        future_idx += 1
                        future_tensor = torch.tensor(episode_entry[trans_idx+future_idx][feature_name])
                        abs_diff = torch.abs(current_tensor - future_tensor)
                        if abs_diff.max().item() > 0:
                            break               
                    trans_entry[new_feature_name] = future_idx
            else:
                raise ValueError(f"Features of type {type(episode_entry[trans_idx])} are not currently supported for {new_feature_name}")
            return episode_entry
    elif mode == "change_loc":
        new_feature_name = f"{feature_name}_change_loc"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            if type(episode_entry[0][feature_name]) == tuple:
                for trans_idx, trans_entry in enumerate(episode_entry):
                    future_idx = 0
                    current_tensor = torch.tensor(episode_entry[trans_idx][feature_name])
                    change_loc = -1
                    while trans_idx + future_idx < episode_length - 2:
                        future_idx += 1
                        future_tensor = torch.tensor(episode_entry[trans_idx+future_idx][feature_name])
                        abs_diff = torch.abs(current_tensor - future_tensor)
                        if abs_diff.max().item() > 0:
                            change_idx = torch.argmax(abs_diff).item()
                            change_loc = episode_entry[trans_idx][feature_name][change_idx]
                            break
                        elif trans_idx + future_idx == episode_length:
                            change_loc = episode_entry[trans_idx][feature_name][change_idx] # need to check this - are indices of boxes consistent? NO THEY ARE NOT - IGNORE THIS
                            break               
                    trans_entry[new_feature_name] = change_loc  # IS NOW -1 FOR UNCHANGED
            else:
                raise ValueError(f"Features of type {type(episode_entry[trans_idx])} are not currently supported for {new_feature_name}")
            return episode_entry
    else:
        raise ValueError(f"User entered mode {mode}, valid modes are: ahead, traj, change")
    return feature_detector


def make_binary_feature_detector(mode: str, feature_name: str, threshold: int) -> Callable:
    new_feature_name = f"{feature_name}_{mode}_{threshold}"
    if mode == "lessthan":
        def binary_feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            assert type(episode_entry[0][feature_name]) is int or type(episode_entry[0][feature_name]) is float,f"Error: This feature detector constructs binary features from ints or floats, {feature_name} is of type {type(episode_entry[0][feature_name])}"
            episode_length = len(episode_entry)
            for trans_entry in episode_entry:
                trans_entry[new_feature_name] = 1 if (trans_entry[feature_name] <= threshold) else 0
            return episode_entry
    elif mode == "equal":
        def binary_feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            assert type(episode_entry[0][feature_name]) is int or type(episode_entry[0][feature_name]) is float,f"Error: This feature detector constructs binary features from ints or floats, {feature_name} is of type {type(episode_entry[0][feature_name])}"
            episode_length = len(episode_entry)
            for trans_entry in episode_entry:
                trans_entry[new_feature_name] = 1 if (trans_entry[feature_name] == threshold) else 0
            return episode_entry
    else:
        raise ValueError("Unsupported mode for make_binary_feature_detector")    
    return binary_feature_detector

def make_trajectory_detector(steps_ahead: int, feature_name: str, mode: str = "default", alt_feature_name: Optional[str] = None, alt_feature_value: Optional[int] = None, inc_current: bool = False) -> Callable:
    assert feature_name in ["agent_loc", "box_loc", "tracked_box_loc_active", "tar_next", "tracked_box_loc_next", "tar_loc", "next_start_move_box_loc", "next_end_move_box_loc", "box1_loc", "box2_loc", "box3_loc", "box4_loc", "tracked_box_loc_change_after_action_1", "tracked_box_loc_change_after_action_2", "tracked_box_loc_change_after_action_3", "tracked_box_loc_change_after_action_4", "tracked_box_loc_change_with_action_1", "tracked_box_loc_change_with_action_2", "tracked_box_loc_change_with_action_3","tracked_box_loc_change_with_action_4"], "Cannot detect trajectory for feature other than agent or box location"
    if mode == "default":
        def get_future_trajectories(episode_entry: list) -> list:
            #print(feature_name)
            if feature_name == "agent_loc" or feature_name == "next_start_move_box_loc":
                virtual_ext = [{"agent_loc": episode_entry[-1]["boxnotontar_loc"]}]
            elif feature_name == "tracked_box_loc_active" or feature_name == "tar_next" or feature_name == "tracked_box_loc_next" or feature_name in ["box1_loc","box2_loc", "box3_loc", "box4_loc", "next_end_move_box_loc", "tracked_box_loc_change_after_action_1", "tracked_box_loc_change_after_action_2", "tracked_box_loc_change_after_action_3", "tracked_box_loc_change_after_action_4"]:
                virtual_ext = [{feature_name: episode_entry[-1]["justtar_loc"]}]
            elif feature_name == "tar_loc" or feature_name == "box_loc":
                virtual_ext = [{feature_name: episode_entry[-1][feature_name]}]
            elif feature_name in ["tracked_box_loc_change_with_action_1", "tracked_box_loc_change_with_action_2", "tracked_box_loc_change_with_action_3", "tracked_box_loc_change_with_action_4"]:
                virtual_ext = [{feature_name: tuple()}]
            for trans_idx, trans in enumerate(episode_entry):
                feature_locs_xy = []
                for future_trans in (episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)]:
                    feature_locs_xy += [(future_trans[feature_name][feature_idx] % 8, (future_trans[feature_name][feature_idx]-(future_trans[feature_name][feature_idx]%8))//8) for feature_idx in range(len(future_trans[feature_name]))]
                feature_locs_xy = torch.tensor(feature_locs_xy)
                trajectory = torch.zeros(size=(8,8), dtype=torch.long)
                if len(feature_locs_xy.shape) != 1:
                    trajectory[feature_locs_xy[:,1],feature_locs_xy[:,0]] = 1
                trans[f"{feature_name}_{'future_trajectory' if steps_ahead!=0 else 'current'}_{steps_ahead}"] = trajectory
            return episode_entry
    elif mode == "conjunction":
        def get_future_trajectories(episode_entry: list) -> list:
            if feature_name == "agent_loc":
                virtual_ext = [{"agent_loc": episode_entry[-1]["boxnotontar_loc"]}]
            elif feature_name == "tracked_box_loc_active" or feature_name == "tar_next" or feature_name == "tracked_box_loc_next" or feature_name == "tracked_box_loc_next" or feature_name in ["box1_loc","box2_loc", "box3_loc", "box4_loc"]:
                virtual_ext = [{feature_name: episode_entry[-1]["justtar_loc"]}]
            elif feature_name == "tar_loc" or feature_name == "box_loc":
                virtual_ext = [{feature_name: episode_entry[-1][feature_name]}]
            virtual_ext[0]["action"] = 0 # dummy no-op action at end of episode
            for trans_idx, trans in enumerate(episode_entry):
                feature_locs_xy = []
                for future_trans in (episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)]:
                    if future_trans[alt_feature_name] == alt_feature_value:
                        feature_locs_xy += [(future_trans[feature_name][feature_idx] % 8, (future_trans[feature_name][feature_idx]-(future_trans[feature_name][feature_idx]%8))//8) for feature_idx in range(len(future_trans[feature_name]))]
                #if type(alt_feature_value) != list:
                    #for future_trans_idx, future_trans in enumerate((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][:-1]):
                        #if future_trans[alt_feature_name] == alt_feature_value:
                            #feature_locs_xy += [((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx] % 8, ((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]-((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]%8))//8) for feature_idx in range(len((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name]))]               
                #else:
                    #for future_trans_idx, future_trans in enumerate((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][:-1]):
                        #for alt_value in alt_feature_value:
                            #if future_trans[alt_feature_name] == alt_value:
                                #feature_locs_xy += [((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx] % 8, ((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]-((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]%8))//8) for feature_idx in range(len((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name]))]                                   
                feature_locs_xy = torch.tensor(feature_locs_xy)
                trajectory = torch.zeros(size=(8,8), dtype=torch.long)
                if len(feature_locs_xy.shape) != 1:
                    trajectory[feature_locs_xy[:,1],feature_locs_xy[:,0]] = 1
                trans[f"{feature_name}_conj_{alt_feature_name}_equals_{alt_feature_value}_{'future_trajectory' if steps_ahead!=0 else 'current'}_{steps_ahead}"] = trajectory
            return episode_entry
    else:
        raise ValueError(f"Unsupported mode: {mode}")

        
    return get_future_trajectories

def generate_aug_trans(episode_entry):
    trans = episode_entry[-1]
    agent_loc = trans["agent_loc"][0]
    agent_loc = ((agent_loc -(agent_loc % 8))//8, agent_loc % 8,)
    agent_y, agent_x = agent_loc
    box_locs = [((box_loc -(box_loc % 8))//8, box_loc % 8) for box_loc in trans["tracked_box_loc"]]
    wall_locs = [((wall_loc-(wall_loc % 8))//8, wall_loc % 8) for wall_loc in trans["board_state"][0].view(-1).topk(k=(trans["board_state"][0]==1).to(int).sum()).indices]
    if ((agent_y-1,agent_x) in box_locs) and trans["action"] == 1:
        if agent_y > 1 and (agent_y-2,agent_x) not in wall_locs and (agent_y-2,agent_x) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y-1,agent_x) else (agent_y-2,agent_x) for box_loc in box_locs]
            new_agent_loc = (agent_y-1,agent_x)
        else:
            new_agent_loc = agent_loc
            new_box_locs = box_locs
    elif ((agent_y+1,agent_x) in box_locs) and trans["action"] == 2:
        if agent_y < 6 and (agent_y+2,agent_x) not in wall_locs and (agent_y+2,agent_x) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y+1,agent_x) else (agent_y+2,agent_x) for box_loc in box_locs]
            new_agent_loc = (agent_y+1,agent_x)
        else:
            new_agent_loc = agent_loc
            new_box_locs = box_locs
    elif ((agent_y,agent_x-1) in box_locs) and trans["action"] == 3:
        if agent_x > 1  and (agent_y,agent_x-2) not in wall_locs and (agent_y,agent_x-2) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y,agent_x-1) else (agent_y,agent_x-2) for box_loc in box_locs]
            new_agent_loc = (agent_y,agent_x-1)
        else:
            new_box_locs = box_locs
            new_agent_loc = agent_loc
    elif ((agent_y,agent_x+1) in box_locs) and trans["action"] == 4:
        if agent_x < 6 and (agent_y,agent_x+2) not in wall_locs and (agent_y,agent_x+2) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y,agent_x+1) else ((agent_y,agent_x+2) if agent_x<6 else (agent_y,agent_x+1)) for box_loc in box_locs]
            new_agent_loc = (agent_y,agent_x+1)
        else:
            new_box_locs = box_locs
            new_agent_loc = agent_loc
    else:
        new_box_locs = box_locs
        if trans["action"] == 1 and agent_y > 0:
            new_agent_loc = (agent_y-1,agent_x)
        elif trans["action"] == 2 and agent_y < 7:
            new_agent_loc = (agent_y+1,agent_x)
        elif trans["action"] == 3 and agent_x > 0:
            new_agent_loc = (agent_y,agent_x-1)
        elif trans["action"] == 4 and agent_y < 7:
            new_agent_loc = (agent_y, agent_x+1)
        else:
            new_agent_loc = agent_loc

    new_box_locs = tuple([(8*y+x) for y,x in new_box_locs])
    new_agent_loc = tuple([8*new_agent_loc[0] + new_agent_loc[1]])
    trans = {"tracked_box_loc": new_box_locs, "agent_loc": new_agent_loc, "action": 0}
    return trans

def make_agent_info_extractor() -> Callable:
    def agent_info_extractor(episode_entry: list) -> list:
        # track squares from which agent performs actions to leave
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx:-1]):
                    if loc_idx in future_trans["agent_loc"] and (future_trans["agent_loc"] != aug_episode_entry[trans_idx+future_trans_idx+1]["agent_loc"]): #NB: ignore no-ops and effective no-ops since want action we leave square with 
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = future_trans["action"]
                        break
            trans["agent_onto_with"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["agent_from"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        # track squares from which agent performs action to enter
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx+1:]):
                    if loc_idx in future_trans["agent_loc"] and aug_episode_entry[trans_idx+future_trans_idx]["agent_loc"] != future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = aug_episode_entry[trans_idx+future_trans_idx]["action"]
                        break
            trans["agent_onto_after"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["agent_onto"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        for trans_idx, trans in enumerate(episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans in episode_entry[trans_idx+1:]:
                    if loc_idx in future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] += (1 if board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] <= 2 else 0)
            trans["agent_loc_count"] = board_locs
        return episode_entry
    return agent_info_extractor

def make_box_info_extractor(unq: bool = False) -> Callable:
    def box_info_extractor(episode_entry: list) -> list:
        # track box_locs
        tracked_box_locs = [episode_entry[0]["box_loc"][i] for i in range(4)]
        episode_entry[0]["tracked_box_loc"] = tuple([tracked_box_locs[i] for i in range(4)])
        for trans in episode_entry[1:]:
            trans_box_locs = trans["box_loc"]
            for i in range(4):
                if tracked_box_locs[i] not in trans_box_locs:
                    for j in range(4):
                        if trans_box_locs[j] not in tracked_box_locs:
                            tracked_box_locs[i] = trans_box_locs[j]
            trans["tracked_box_loc"] = tuple([tracked_box_locs[i] for i in range(4)])

        # get final position of boxes
        tracked_box_locs_final = [box_loc for box_loc in episode_entry[-1]["tracked_box_loc"]]
        for k in range(4):
            if tracked_box_locs_final[k] not in episode_entry[-1]["tar_loc"]:
                for s in range(4):
                    if episode_entry[-1]["tar_loc"][s] not in tracked_box_locs_final:
                        tracked_box_locs_final[k] = episode_entry[-1]["tar_loc"][s]

        # track next box to move
        for trans_idx, trans in enumerate(episode_entry):
            current_box_locs = trans["tracked_box_loc"]
            i = 0
            while True:
                if current_box_locs != episode_entry[trans_idx+i]["tracked_box_loc"]:
                    changed_box_locs = episode_entry[trans_idx+i]["tracked_box_loc"]
                    break
                elif trans_idx+i==len(episode_entry)-1:
                    changed_box_locs = tracked_box_locs_final
                    break
                else:
                    i += 1
            for j in range(4):
                if current_box_locs[j] != changed_box_locs[j]:
                    trans["tracked_box_loc_next_idx"] = j
            trans["tracked_box_loc_next"] = (trans["tracked_box_loc"][trans["tracked_box_loc_next_idx"]],)

        # track if box moves for rest of episode, and track active boxes:
        episode_entry[-1]["box_loc_final"] = tuple([1 if episode_entry[-1]["tracked_box_loc"][i] in tracked_box_locs_final else 0 for i in range(4)])
        for i in range(len(episode_entry)-1):
            current_box_locs = episode_entry[-(i+2)]["tracked_box_loc"]
            future_box_locs = episode_entry[-(i+1)]["tracked_box_loc"]
            future_box_locs_final = episode_entry[-(i+1)]["box_loc_final"]
            current_box_locs_final = tuple([1 if (current_box_locs[j]==future_box_locs[j] and future_box_locs_final[j]==1) else 0 for j in range(4)])
            episode_entry[-(i+2)]["box_loc_final"] = current_box_locs_final
        for trans in episode_entry:
            trans["tracked_box_loc_active"] = tuple([trans["tracked_box_loc"][i] for i in range(4) if trans["box_loc_final"][i]==0])
        if unq:
            for trans in episode_entry:
                next_box = trans["tracked_box_loc_next"]
                if next_box == trans["box1_loc"]:
                    trans["tracked_box_loc_next_type"] = 0
                elif next_box == trans["box2_loc"]:
                    trans["tracked_box_loc_next_type"] = 1
                elif next_box == trans["box3_loc"]:
                    trans["tracked_box_loc_next_type"] = 2
                elif next_box == trans["box4_loc"]:
                    trans["tracked_box_loc_next_type"] = 3
                else:
                    raise ValueError("Could not match tracked_box_loc_next to the location of a known box")
                
        # track number of turns since each tracked box last moved
        tracked_box_locs = episode_entry[0]["tracked_box_loc"]
        episode_entry[0]["tracked_box_loc_since_last_moved"] = (42, 42, 42, 42)
        old_last_moved = episode_entry[0]["tracked_box_loc_since_last_moved"]
        for trans_idx, trans in enumerate(episode_entry):
            current_box_locs = trans["tracked_box_loc"]
            trans["tracked_box_loc_since_last_moved"] = tuple([old_last_moved[i]+1 if current_box_locs[i]==tracked_box_locs[i] else 0 for i in range(4)])
            old_last_moved = trans["tracked_box_loc_since_last_moved"]
            if current_box_locs != tracked_box_locs:
                tracked_box_locs = current_box_locs

        # track square from which agent next pushes box with >1 step gap between pushing it
        for trans_idx, trans in enumerate(episode_entry):
            if trans_idx == len(episode_entry)-1:
                if len(episode_entry) > 114: # 
                    trans["start_move_box"] = 0 # kind of a weird one: if level is unsolved, then *empirically* agent paces and never seems to move another box
                elif sum([1 if trans["tracked_box_loc_since_last_moved"][i] != 0 else 0 for i in range(4)]) != 0:
                    trans["start_move_box"] = 1
                else:
                    trans["start_move_box"] = 0
            elif sum([1 if (episode_entry[trans_idx+1]["tracked_box_loc_since_last_moved"][i] == 0 and trans["tracked_box_loc_since_last_moved"][i] != 0) else 0 for i in range(4)]) != 0:
                trans["start_move_box"] = 1
            else:
                trans["start_move_box"] = 0
        for trans_idx, trans in enumerate(episode_entry):
            i = 0
            while True:
                if episode_entry[trans_idx+i]["start_move_box"] == 1:
                    trans["next_start_move_box_loc"] = episode_entry[trans_idx+i]["agent_loc"]
                    break
                elif trans_idx+i == len(episode_entry)-1:
                    trans["next_start_move_box_loc"] = tuple([-1]) # if agent doesn't move a box for the rest of the episode, no feature
                    break
                else:
                    i += 1

        # track next square which agent pushes box to an leaves for >1 square
        for trans_idx, trans in enumerate(episode_entry):
            if trans_idx == len(episode_entry)-1:
                trans["end_box_move"] = 0 # if episode terminates without solving, agent paces and will not be pushing box on final step; if next turn is termination, agent cannot push and then take a break
            elif sum([1 if (trans["tracked_box_loc_since_last_moved"][i] == 0 and episode_entry[trans_idx+1]["tracked_box_loc_since_last_moved"][i] != 0) else 0 for i in range(4)]) != 0:
                trans["end_box_move"] = 1
            else:
                trans["end_box_move"] = 0
        for trans_idx, trans in enumerate(episode_entry):
            i = 1
            while True:
                if trans_idx+i >= len(episode_entry):
                    if len(episode_entry) > 114:
                        trans["next_end_move_box_loc"] = tuple([-1]) # if episode isn't solved, assume final step is not pushing a box for future movement (makes sense)
                    else:
                        trans["next_end_move_box_loc"] = tuple([trans["justtar_loc"][0]]) # if final step in solved level, will just be pushing box onto target
                    break
                elif episode_entry[trans_idx+i]["end_box_move"] == 1:
                    trans["next_end_move_box_loc"] = tuple([[episode_entry[trans_idx+i]["tracked_box_loc"][j] for j in range(len(trans["tracked_box_loc"])) if trans["tracked_box_loc"][j]!=episode_entry[trans_idx+i]["tracked_box_loc"][j]][0]])
                    break
                else:
                    i += 1

        # track locations where box is pushed *after* action X on prior turn
        episode_entry[0]["tracked_box_loc_change_after_action_1"] = tuple()
        episode_entry[0]["tracked_box_loc_change_after_action_2"] = tuple()
        episode_entry[0]["tracked_box_loc_change_after_action_3"] = tuple()
        episode_entry[0]["tracked_box_loc_change_after_action_4"] = tuple()
        aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry[1:]):
            trans["tracked_box_loc_change_after_action_1"] = tuple()
            trans["tracked_box_loc_change_after_action_2"] = tuple()
            trans["tracked_box_loc_change_after_action_3"] = tuple()
            trans["tracked_box_loc_change_after_action_4"] = tuple()
            if trans["tracked_box_loc"] != aug_episode_entry[trans_idx]["tracked_box_loc"]:
                for i in range(4):
                    if trans["tracked_box_loc"][i] != aug_episode_entry[trans_idx]["tracked_box_loc"][i]:
                        trans[f"tracked_box_loc_change_after_action_{aug_episode_entry[trans_idx]['action']}"] = tuple([trans["tracked_box_loc"][i]])
        #episode_entry = aug_episode_entry[:-1]

        # track locations where box is pushed *with* action X on current turn
        aug_episode_entry[0]["tracked_box_loc_change_with_action_1"] = tuple()
        aug_episode_entry[0]["tracked_box_loc_change_with_action_2"] = tuple()
        aug_episode_entry[0]["tracked_box_loc_change_with_action_3"] = tuple()
        aug_episode_entry[0]["tracked_box_loc_change_with_action_4"] = tuple()
        #aug_episode_entry = episode_entry + [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry[1:]):
            trans["tracked_box_loc_change_with_action_1"] = tuple()
            trans["tracked_box_loc_change_with_action_2"] = tuple()
            trans["tracked_box_loc_change_with_action_3"] = tuple()
            trans["tracked_box_loc_change_with_action_4"] = tuple()
            if trans["tracked_box_loc"] != aug_episode_entry[trans_idx]["tracked_box_loc"]:
                for i in range(4):
                    if trans["tracked_box_loc"][i] != aug_episode_entry[trans_idx]["tracked_box_loc"][i]:
                        aug_episode_entry[trans_idx][f"tracked_box_loc_change_with_action_{aug_episode_entry[trans_idx]['action']}"] = tuple([aug_episode_entry[trans_idx]["tracked_box_loc"][i]])
        #episode_entry = aug_episode_entry[:-1]
         
        # track the direction from which a box is next pushed onto this square (0 if no box is pushed onto this square for the rest of the episode)
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans in aug_episode_entry[trans_idx+1:]:
                    if loc_idx in future_trans["tracked_box_loc_change_after_action_1"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 1
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_after_action_2"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 2
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_after_action_3"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 3
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_after_action_4"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 4
                        break
            trans["tracked_box_next_push_onto_after"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["tracked_box_next_push_onto"] = new_board_locs

        # track the direction in which a box is next pushed from this square (0 if no box is pushed onto this square for the rest of the episode)
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((8,8), dtype=int)
            for loc_idx in range(64):
                for future_trans in aug_episode_entry[trans_idx:]:
                    if loc_idx in future_trans["tracked_box_loc_change_with_action_1"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 1
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_with_action_2"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 2
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_with_action_3"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 3
                        break
                    elif loc_idx in future_trans["tracked_box_loc_change_with_action_4"]:
                        board_locs[(loc_idx-loc_idx%8)//8, loc_idx%8] = 4
                        break
            trans["tracked_box_next_push_onto_with"] = board_locs
            new_board_locs = torch.zeros((8,8), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans["tracked_box_next_push_from"] = new_board_locs
        episode_entry = aug_episode_entry[:-1]
        return episode_entry
    return box_info_extractor

def make_tar_info_extractor(unq: bool = False) -> Callable:
    def tar_info_extractor(episode_entry: list) -> list:
        # track state of each target
        for trans in episode_entry:
            tar_states = [0, 0, 0, 0]
            for i in range(4):
                if trans["tar_loc"][i] in trans["boxontar_loc"]:
                    tar_states[i] = 1
            trans["tar_state"] = tuple(tar_states)
        # track if each target in its final state
        episode_entry[-1]["tar_state_final"] = tuple([1 if episode_entry[-1]["tar_state"][i]==1 else 0 for i in range(4)])
        for i in range(len(episode_entry)-1):
            current_tar_locs = episode_entry[-(i+2)]["tar_state"]
            future_tar_locs = episode_entry[-(i+1)]["tar_state"]
            future_tar_locs_final = episode_entry[-(i+1)]["tar_state_final"]
            current_tar_locs_final = tuple([1 if (current_tar_locs[j]==future_tar_locs[j] and future_tar_locs_final[j]==1) else 0 for j in range(4)])
            episode_entry[-(i+2)]["tar_state_final"] = current_tar_locs_final
        # track next box to move
        for trans_idx, trans in enumerate(episode_entry):
            current_tar_locs = trans["tar_state"]
            i = 0
            while True:
                if current_tar_locs != episode_entry[trans_idx+i]["tar_state"]:
                    changed_tar_locs = episode_entry[trans_idx+i]["tar_state"]
                    changed_tar_idx = [i for i in range(4) if changed_tar_locs[i]!=current_tar_locs[i]][0]
                    if sum([1 for future_trans in episode_entry[trans_idx+i+1:] if future_trans["tar_state"][changed_tar_idx]!=changed_tar_locs[changed_tar_idx]]) == 0:
                        break
                    else:
                        i += 1
                elif trans_idx+i==len(episode_entry)-1:
                    changed_tar_locs = tuple([1, 1, 1, 1])
                    break
                else:
                    i += 1
            for j in range(4):
                if current_tar_locs[j] != changed_tar_locs[j]:
                    trans["tar_next_idx"] = j
            trans["tar_next"] = (trans["tar_loc"][trans["tar_next_idx"]],)
        if unq:
            tar1_loc = episode_entry[0]["tar1_loc"] if episode_entry[0]["tar1_loc"] else episode_entry[1]["tar1_loc"]
            tar2_loc = episode_entry[0]["tar2_loc"] if episode_entry[0]["tar2_loc"] else episode_entry[1]["tar2_loc"]
            tar3_loc = episode_entry[0]["tar3_loc"] if episode_entry[0]["tar3_loc"] else episode_entry[1]["tar3_loc"]
            tar4_loc = episode_entry[0]["tar4_loc"] if episode_entry[0]["tar4_loc"] else episode_entry[1]["tar4_loc"]
            for trans_idx, trans in enumerate(episode_entry):
                next_tar = trans["tar_next"]
                if next_tar == tar1_loc:
                    trans["tar_next_type"] = 0
                elif next_tar == tar2_loc:
                    trans["tar_next_type"] = 1
                elif next_tar == tar3_loc:
                    trans["tar_next_type"] = 2
                elif next_tar == tar4_loc:
                    trans["tar_next_type"] = 3
                else:
                    print(next_tar, trans["tar_loc"], trans["tar1_loc"], trans["tar2_loc"], trans["tar3_loc"], trans["tar4_loc"], trans["boxontar_loc"], trans["agent_loc"], tar1_loc, tar2_loc, tar3_loc, tar4_loc)
                    raise ValueError("Could not match next_tar to the location of a known target")
        return episode_entry
    return tar_info_extractor

@torch.no_grad()
def create_probing_data(drc_net: DRCNet, env: Env, flags: NamedTuple, num_episodes: int, current_board_feature_fncs: list, future_feature_fncs: list,
                         binary_feature_fncs: list, prob_accept: float = 1.0, debug: bool = False) -> list:
    """Generate a list where each entry is a dictionary of features corresponding to a single transition

    Args:
        drc_net (DRCNet): Trained DRC network used to generate transitions
        env (Env): Sokoban environment
        flags (NamedTuple): flag object
        num_episodes (int): number of episodes to run to generate the transitions
        current_board_feature_fncs (list): list of tuples of the form (feature_name, feature_fnc), where each feature_fnc extracts a discrete feature from the current state of the Sokoban board; this feature is then added to the episode entry (dictionary) with the key feature_name
        future_feature_fncs (list): list of functions where each function adds a feature to the current transition corresponding to the value taken by some other feature in a future transition
        prob_accept (float): probability that each transition entry is independently accepted into the dataset

    Returns:
        list: returns probing_data, a list of dictionaries where each dictionary contains features for a single transition generated by the DRC agent
    """
    bl_drc_net = DRCNet(
            obs_space=env.observation_space,
            action_space=env.action_space,
            flags=flags,
            record_state=True,
        )

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

        state, reward, done, info = env.step(actor_out.action)
        trans_entry["reward"] = round(reward.item(), 3) # round rewards to 3 d.p.
        episode_entry.append(trans_entry)

        if done:
            for fnc in future_feature_fncs:
                episode_entry = fnc(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry["steps_remaining"] = episode_length - trans_idx
                trans_entry["steps_taken"] = trans_idx+1
                trans_entry["return"] = sum([(0.97**t)*future_trans["reward"] for t, future_trans in enumerate(episode_entry[trans_idx:])])

            for fnc in binary_feature_fncs:
                episode_entry = fnc(episode_entry)
            
            
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

    return probing_data

def make_selector(mode: str, feature_name: str = "agent_loc", threshold: int = 1, prob_accept: float = 1.0) -> list:
    if mode == "random":
        def selector(probing_data: list) -> list:
            pruned_data = []
            for trans_entry in probing_data:
                if prob_accept > uniform(0,1):
                    pruned_data.append(trans_entry)
            return pruned_data
    elif mode == "lessthan":
        def selector(probing_data: list) -> list:
            pruned_data = []
            for trans_entry in probing_data:
                if trans_entry[feature_name] <= threshold and prob_accept > uniform(0,1):
                    pruned_data.append(trans_entry)
            return pruned_data
    elif mode == "greaterthan":
        def selector(probing_data: list) -> list:
            pruned_data = []
            for trans_entry in probing_data:
                if trans_entry[feature_name] >= threshold and prob_accept > uniform(0,1):
                    pruned_data.append(trans_entry)
            return pruned_data
    else:
        raise ValueError(f"no such mode as {mode} supported by make_selector")
    return selector

class ProbingDataset(Dataset):
    def __init__(self, data: list):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, index: int) -> dict:
        return self.data[index]
    def get_feature_range(self, feature: str) -> tuple[int, int]:
        assert feature in self.data[0].keys(), f"Please enter a feature in dataset: {self.data[0].keys()}"
        min_feature_value, max_feature_value = self.data[0][feature], self.data[0][feature]
        for entry in self.data:
            if entry[feature] > max_feature_value:
                max_feature_value = entry[feature]
            elif entry[feature] < min_feature_value:
                min_feature_value = entry[feature]
        return (min_feature_value, max_feature_value)


class ProbingDatasetCleaned(Dataset):
    def __init__(self, data: list):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, index: int) -> tuple:
        return self.data[index]
        

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="run convprobe patching exps")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="250m")
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--unq", type=bool, default=False)
    parser.add_argument("--pct_train", type=float, default=1)
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--mini", type=bool, default=True)
    parser.add_argument("--name", type=str, default="train")
    args = parser.parse_args()

    mini = args.mini
    name = args.name
    gpu = args.gpu
    pct_train = args.pct_train
    unq = args.unq
    num_episodes = args.num_episodes
    debug = args.debug
    model_name = args.model_name
    env_name = args.env_name

    if unq:
        #env = make("Sokoban-v0",env_n=1,gpu=gpu,wrapper_type=1,has_model=False,train_model=False,parallel=False,save_flags=False,mini=mini)
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
            mini_unqtar=True,
            mini_unqbox=True         
            ) 
        flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
        flags.mini = True
        flags.mini_unqtar = True
        flags.mini_unqbox = True
        drc_net = DRCNet(
            obs_space=env.observation_space,
            action_space=env.action_space,
            flags=flags,
            record_state=True,

        )
        ckp_path = "../drc_unq"
        ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{model_name}.tar")

        adj_wall_detector = make_current_board_feature_detector(feature_idxs=[0], mode="adj")
        adj_boxnotontar_detector = make_current_board_feature_detector(feature_idxs=[2,10,11,12], mode="adj")
        adj_boxontar_detector = make_current_board_feature_detector(feature_idxs=[3], mode="adj")
        adj_box_detector = make_current_board_feature_detector(feature_idxs=[2,3,10,11,12], mode="adj")
        adj_tar_detector = make_current_board_feature_detector(feature_idxs=[6,7,8,9], mode="adj")
        num_boxnotontar_detector = make_current_board_feature_detector(feature_idxs=[2,10,11,12], mode="num")
        agent_loc_detector = make_current_board_feature_detector(feature_idxs=[4,5], mode="loc")
        box_loc_detector = make_current_board_feature_detector(feature_idxs=[2,3,10,11,12], mode="loc")
        tar_loc_detector = make_current_board_feature_detector(feature_idxs=[3,5,6,7,8,9], mode="loc")
        boxontar_loc_detector = make_current_board_feature_detector(feature_idxs=[3], mode="loc")
        justtar_loc_detector = make_current_board_feature_detector(feature_idxs=[5,6], mode="loc")
        boxnotontar_loc_detector = make_current_board_feature_detector(feature_idxs=[2,10,11,12], mode="loc")

    else:
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
        ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{model_name}.tar")

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

    if unq:
        current_board_feature_fncs +=[
            ("box1_loc", make_current_board_feature_detector(feature_idxs=[2], mode="loc")),
            ("box2_loc", make_current_board_feature_detector(feature_idxs=[10], mode="loc")),
            ("box3_loc", make_current_board_feature_detector(feature_idxs=[11], mode="loc")),
            ("box4_loc", make_current_board_feature_detector(feature_idxs=[12], mode="loc")),
            ("tar1_loc", make_current_board_feature_detector(feature_idxs=[6], mode="loc")),
            ("tar2_loc", make_current_board_feature_detector(feature_idxs=[7], mode="loc")),
            ("tar3_loc", make_current_board_feature_detector(feature_idxs=[8], mode="loc")),
            ("tar4_loc", make_current_board_feature_detector(feature_idxs=[9], mode="loc"))
        ]

    #future_feature_fncs = [make_future_feature_detector(feature_name="action",steps_ahead=t, mode="ahead") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="reward",steps_ahead=t, mode="ahead") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="value",steps_ahead=t, mode="ahead") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="agent_loc",steps_ahead=t, mode="ahead") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="box_loc",steps_ahead=t, mode="ahead") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="value",steps_ahead=t, mode="ahead") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="action",steps_ahead=t, mode="traj") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="reward",steps_ahead=t, mode="traj") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="value",steps_ahead=t, mode="traj") for t in range(1,4)]
    #future_feature_fncs += [make_future_feature_detector(feature_name="num_boxnotontar", mode="change")]
    #future_feature_fncs += [make_future_feature_detector(feature_name="action", mode="change")]
    #future_feature_fncs += [make_future_feature_detector(feature_name="box_loc", mode="change")]
    #future_feature_fncs += [make_future_feature_detector(feature_name="box_loc", mode="change_loc")]
    future_feature_fncs = [make_tar_info_extractor(unq=unq), make_box_info_extractor(unq=False), make_agent_info_extractor()]

    #future_feature_fncs += [make_trajectory_detector(feature_name="agent_loc", steps_ahead=i) for i in [1,5,10,20,120]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="agent_loc", steps_ahead=i, mode="conjunction", alt_feature_name="action", alt_feature_value=j) for i in [120] for j in [1,2,3,4]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="agent_loc", steps_ahead=i, mode="conjunction", alt_feature_name="action", alt_feature_value=[j,k]) for i in [120] for j in [1,2,3,4] for k in [1,2,3,4] if j>k]
    #future_feature_fncs += [make_trajectory_detector(feature_name="box_loc", steps_ahead=i) for i in [1,5,10,20,120]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tracked_box_loc_active", steps_ahead=i) for i in [1,5,10,20,120]]
    #future_feature_fncs += [make_trajectory_detector(feature_name=f"tracked_box_loc_change_after_action_{j}", steps_ahead=i) for i in [10,20,120] for j in [1,2,3,4]]
    #future_feature_fncs += [make_trajectory_detector(feature_name=f"tracked_box_loc_change_with_action_{j}", steps_ahead=i) for i in [10,20,120] for j in [1,2,3,4]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tracked_box_loc_next", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tar_next", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tar_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="box_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="agent_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tracked_box_loc_active", steps_ahead=0, inc_current=True)]

    #future_feature_fncs += [make_trajectory_detector(feature_name="next_end_move_box_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="next_end_move_box_loc", steps_ahead=i) for i in [10,20,120]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="next_start_move_box_loc", steps_ahead=0, inc_current=True)]

    #if unq:
        #future_feature_fncs += [make_trajectory_detector(feature_name=f"box{j}_loc", steps_ahead=i) for i in [1,5,10,20,120] for j in [1,2,3,4]] 
    
    


    probing_data = create_probing_data(drc_net=drc_net,
                                       env=env,
                                       flags=flags,
                                       num_episodes=num_episodes,
                                       current_board_feature_fncs=current_board_feature_fncs,
                                       future_feature_fncs=future_feature_fncs,
                                       binary_feature_fncs=[], 
                                       debug=debug)
    
    if debug:
        for trans in probing_data[:250]: # check that h,c,x_enc correctly ordered by ensuring decoding with policy head behaves as expected
            x = trans["hidden_states"].to(env.device)
            core_output = x[-1,64*2:64*2+32,:,:]
            x_enc = x[-1,192:224,:,:]
            core_output = torch.cat([x_enc, core_output], dim=0)
            core_output = torch.flatten(core_output).view(1,-1)
            final_out = relu(drc_net.final_layer(core_output))
            pri_logits = drc_net.policy(final_out)
            assert torch.argmax(pri_logits, dim=-1).item() == trans["action"], "hidden states are incorrectly ordered - decoding [h,x_enc] with the policy head does not produce the chosen action as expected"

    final_train_board = int(num_episodes * pct_train)
    final_val_board = final_train_board + int(num_episodes * round(0.5 * (1 - pct_train), 2))
    probing_train_data = [entry for entry in probing_data if entry["board_num"] <= final_train_board]
    probing_val_data = [entry for entry in probing_data if entry["board_num"] > final_train_board and entry["board_num"] <= final_val_board]
    probing_test_data = [entry for entry in probing_data if entry["board_num"] > final_val_board]
    
    print(f"Full train, val and test sets contain {len(probing_train_data)}, {len(probing_val_data)}, {len(probing_test_data)} transitions respectively")
    
    if not debug:
        torch.save(ProbingDataset(probing_train_data), f"./data/{name}_data_full_{model_name}.pt")
        #torch.save(ProbingDataset(probing_val_data), f"./data/val_data_full_{model_name}.pt")
        #torch.save(ProbingDataset(probing_test_data), f"./data/test_data_full_{model_name}.pt")

    selectors = [
            ]
        #("adjbox", make_selector(mode="greaterthan", feature_name="adj_box", threshold=1, prob_accept=0.2)),
        #("noadjbox", make_selector(mode="lessthan", feature_name="adj_box", threshold=0, prob_accept=0.2)),
        #("soon5", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=5, prob_accept=0.3)),
        #("soon4", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=4, prob_accept=0.3)),
        #("soon3", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=3, prob_accept=1)),
        #("soon2", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=2, prob_accept=1)),
        #("soon1", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=1, prob_accept=1)),
        #("start3", make_selector(mode="lessthan", feature_name="steps_taken", threshold=3)),
        #("start2", make_selector(mode="lessthan", feature_name="steps_taken", threshold=2)),
        #("start1", make_selector(mode="lessthan", feature_name="steps_taken", threshold=1)),
        #("onebox", make_selector(mode="lessthan", feature_name="num_boxnotontar", threshold=1, prob_accept=0.2))

