from thinker.main import make, Env
from thinker.actor_net import DRCNet
from torch.utils.data.dataset import Dataset
import torch
from torch.nn.functional import relu
from thinker import util
from typing import Callable, NamedTuple, Optional
from numpy.random import uniform
import os

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
                    while trans_idx + future_idx < episode_length - 2:
                        future_idx += 1
                        future_tensor = torch.tensor(episode_entry[trans_idx+future_idx][feature_name])
                        abs_diff = torch.abs(current_tensor - future_tensor)
                        if abs_diff.max().item() > 0:
                            change_idx = torch.argmax(abs_diff).item()
                            change_loc = episode_entry[trans_idx][feature_name][change_idx]
                            break
                        elif trans_idx + future_idx == episode_length:
                            change_loc = episode_entry[trans_idx][feature_name][change_idx] # need to check this - are indices of boxes consistent?
                            break               
                    trans_entry[new_feature_name] = change_loc if len(episode_entry) > 1 else -1 # need to check this
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

@torch.no_grad()
def create_probing_data(drc_net: DRCNet, env: Env, flags: NamedTuple, num_episodes: int, current_board_feature_fncs: list, future_feature_fncs: list, binary_feature_fncs: list, prob_accept: float = 1.0, debug: bool = False) -> list:
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
    rnn_state = drc_net.initial_state(batch_size=1, device=env.device)
    state = env.reset() 
    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
    actor_out, rnn_state = drc_net(env_out, rnn_state)

    episode_length = 0
    board_num = 0
    probing_data = []
    episode_entry = []
    while(board_num < num_episodes):
        if episode_length > 0:
            trans_entry["reward"] = round(reward.item(), 3) # round rewards to 3 d.p.
            episode_entry.append(trans_entry)
        state, reward, done, info = env.step(actor_out.action)
        env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
        actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)

        if debug:
            print(actor_out.pri_param.argmax(dim=-1).item(), actor_out.action.item())

        trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
        trans_entry["action"] = actor_out.action.item()
        trans_entry["value"] = round(actor_out.baseline.item(), 3) 
        trans_entry["board_state"] = state["real_states"][0] # tensor of size (channels, board_height, board_width)
        trans_entry["hidden_states"] = drc_net.hidden_state[0] # tensor of size (ticks+1, layers*64, representation_height, representation_width)
        trans_entry["board_num"] = board_num
        episode_length += 1

        if done:
            for fnc in future_feature_fncs:
                episode_entry = fnc(episode_entry)

            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry["steps_remaining"] = episode_length - trans_idx - 2
                trans_entry["steps_taken"] = trans_idx

            if len(episode_entry) > 0:
                for fnc in binary_feature_fncs:
                    episode_entry = fnc(episode_entry)

            probing_data += episode_entry
            episode_length = 0
            board_num += 1
            print("Data collected from episode", board_num, "with pruned episode length of", len(episode_entry))
            episode_entry = []

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

    mini = True
    gpu = False
    pct_train = 0.8
    num_episodes = 50
    debug = False

    adj_wall_detector = make_current_board_feature_detector(feature_idxs=[0], mode="adj")
    adj_boxnotontar_detector = make_current_board_feature_detector(feature_idxs=[2], mode="adj")
    adj_boxontar_detector = make_current_board_feature_detector(feature_idxs=[3], mode="adj")
    adj_box_detector = make_current_board_feature_detector(feature_idxs=[2,3], mode="adj")
    adj_tar_detector = make_current_board_feature_detector(feature_idxs=[6], mode="adj")
    num_boxnotontar_detector = make_current_board_feature_detector(feature_idxs=[2], mode="num")
    agent_loc_detector = make_current_board_feature_detector(feature_idxs=[4,5], mode="loc")
    box_loc_detector = make_current_board_feature_detector(feature_idxs=[2,3], mode="loc")
    current_board_feature_fncs = [
        ("adj_walls", adj_wall_detector),
        ("adj_boxnotontar", adj_boxnotontar_detector),
        ("adj_boxontar", adj_boxontar_detector),
        ("adj_box", adj_box_detector),
        ("adj_tar", adj_tar_detector),
        ("num_boxnotontar", num_boxnotontar_detector),
        ("agent_loc", agent_loc_detector),
        ("box_loc", box_loc_detector)
    ]

    future_feature_fncs = [make_future_feature_detector(feature_name="action",steps_ahead=t, mode="ahead") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="reward",steps_ahead=t, mode="ahead") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="value",steps_ahead=t, mode="ahead") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="agent_loc",steps_ahead=t, mode="ahead") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="box_loc",steps_ahead=t, mode="ahead") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="value",steps_ahead=t, mode="ahead") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="action",steps_ahead=t, mode="traj") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="reward",steps_ahead=t, mode="traj") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="value",steps_ahead=t, mode="traj") for t in range(1,4)]
    future_feature_fncs += [make_future_feature_detector(feature_name="num_boxnotontar", mode="change")]
    future_feature_fncs += [make_future_feature_detector(feature_name="action", mode="change")]
    future_feature_fncs += [make_future_feature_detector(feature_name="box_loc", mode="change")]
    future_feature_fncs += [make_future_feature_detector(feature_name="box_loc", mode="change_loc")]


    binary_feature_fncs = [make_binary_feature_detector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=1),
                           make_binary_feature_detector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=3),
                           make_binary_feature_detector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=5),
                           make_binary_feature_detector(mode="lessthan", feature_name="steps_remaining", threshold=1),
                           make_binary_feature_detector(mode="lessthan", feature_name="steps_remaining", threshold=3),
                           make_binary_feature_detector(mode="lessthan", feature_name="steps_remaining", threshold=5),
                           make_binary_feature_detector(mode="lessthan", feature_name="action_until_change", threshold=0),
                           make_binary_feature_detector(mode="equal", feature_name="action", threshold=0),
                           make_binary_feature_detector(mode="equal", feature_name="action", threshold=1),
                           make_binary_feature_detector(mode="equal", feature_name="action", threshold=2),
                           make_binary_feature_detector(mode="equal", feature_name="action", threshold=3),
                           make_binary_feature_detector(mode="equal", feature_name="action", threshold=4)]


    env = make("Sokoban-v0",env_n=1,gpu=gpu,wrapper_type=1,has_model=False,train_model=False,parallel=False,save_flags=False,mini=mini)
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    flags.mini = mini
    drc_net = DRCNet(
        obs_space=env.observation_space,
        action_space=env.action_space,
        flags=flags,
        record_state=True,
    )
    ckp_path = "../drc_mini"
    ckp_path = os.path.join(util.full_path(ckp_path), "ckp_actor_realstep249000192.tar")
    ckp = torch.load(ckp_path, env.device)
    drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    drc_net.to(env.device)

    probing_data = create_probing_data(drc_net=drc_net,
                                       env=env,
                                       flags=flags,
                                       num_episodes=num_episodes,
                                       current_board_feature_fncs=current_board_feature_fncs,
                                       future_feature_fncs=future_feature_fncs,
                                       binary_feature_fncs=binary_feature_fncs,
                                       prob_accept=0.2, 
                                       debug=debug)
    
    for trans in probing_data[:250]: # check that h,c,x_enc correctly ordered by ensuring decoding with policy head behaves as expected
        x = trans["hidden_states"]
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
        torch.save(ProbingDataset(probing_train_data), "./data/train_data_full.pt")
        torch.save(ProbingDataset(probing_val_data), "./data/val_data_full.pt")
        torch.save(ProbingDataset(probing_test_data), "./data/test_data_full.pt")

    selectors = [
        ("random", make_selector(mode="random", prob_accept=0.2)),
        ("adjbox", make_selector(mode="greaterthan", feature_name="adj_box", threshold=1, prob_accept=0.2)),
        ("noadjbox", make_selector(mode="lessthan", feature_name="adj_box", threshold=0, prob_accept=0.2)),
        ("soon5", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=5, prob_accept=0.3)),
        ("soon4", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=4, prob_accept=0.3)),
        ("soon3", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=3, prob_accept=1)),
        ("soon2", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=2, prob_accept=1)),
        ("soon1", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=1, prob_accept=1)),
        ("start3", make_selector(mode="lessthan", feature_name="steps_taken", threshold=3)),
        ("start2", make_selector(mode="lessthan", feature_name="steps_taken", threshold=2)),
        ("start1", make_selector(mode="lessthan", feature_name="steps_taken", threshold=1)),
        ("onebox", make_selector(mode="lessthan", feature_name="num_boxnotontar", threshold=1, prob_accept=0.2))
    ]

    for subset_name, subset_selector in selectors:
        subset_train = subset_selector(probing_train_data)
        subset_val = subset_selector(probing_val_data)
        subset_test = subset_selector(probing_test_data)
        print(f"{subset_name} train, val and test sets contain {len(subset_train)}, {len(subset_val)}, {len(subset_test)} transitions respectively")
        if not debug:
            torch.save(ProbingDataset(subset_train), f"./data/train_data_{subset_name}.pt")
            torch.save(ProbingDataset(subset_val), f"./data/val_data_{subset_name}.pt")
            torch.save(ProbingDataset(subset_test), f"./data/test_data_{subset_name}.pt")
