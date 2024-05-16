from main import make
from actor_net import DRCNet
from torch.utils.data.dataset import Dataset
import torch
import util

def make_feature_detector(feature_idx, mode):
    """Create feature detector functions to extract features from mini-sokoban boards. Boards must be (7,8,8) arrays
    Args:
        feature_idx (int): index of feature of interest (see sokoban.cpp)
        mode (str): either "adj" (to count number of adjacent features) or "num" (to count total number of features on board)
    """
    assert mode in ["adj", "num"], "Please enter a valid mode - either ADJ or NUM"
    if mode == "adj":
        def feature_detector(board):
            h, w = board.shape[1:]
            x, y = ((board[4,:,:]==1) + (board[5,:,:]==1)).nonzero()[0,:]
            adj_coords = [(xp, yp) for xp, yp in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if xp>-1 and xp<h and yp>-1 and yp<w]
            n_hits = 0
            for (xp,yp) in adj_coords:
                if board[feature_idx, xp, yp] == 1:
                    n_hits += 1
            return n_hits
    else:
        def feature_detector(board):
            return torch.sum((board[feature_idx,:,:]==1).int()).item()
    return feature_detector

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

if __name__=="__main__":

    mini = True
    gpu = True
    pct_train = 0.9
    num_episodes = 600

    adj_wall_detector = make_feature_detector(feature_idx=0, mode="adj")
    adj_boxnotontar_detector = make_feature_detector(feature_idx=2, mode="adj")
    adj_boxontar_detector = make_feature_detector(feature_idx=3, mode="adj")
    adj_tar_detector = make_feature_detector(feature_idx=6, mode="adj")
    num_boxnotontar_detector = make_feature_detector(feature_idx=2, mode="num")
    feature_fncs = [
        ("adj_walls", adj_wall_detector),
        ("adj_boxnotontar", adj_boxnotontar_detector),
        ("adj_boxontar", adj_boxontar_detector),
        ("adj_tar_detector", adj_tar_detector),
        ("num_boxnotontar_detector", num_boxnotontar_detector)
    ]

    probing_data = []
    episode_entry = []

    env = make("Sokoban-v0",env_n=1,gpu=gpu,wrapper_type=1,has_model=False,train_model=False,parallel=False,save_flags=False,mini=mini)

    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    flags.mini = mini
    drc_net = DRCNet(
        obs_space=env.observation_space,
        action_space=env.action_space,
        flags=flags,
        record_state=True,
    )
    drc_net.to(env.device)

    rnn_state = drc_net.initial_state(batch_size=1, device=env.device)
    state = env.reset() 
    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)
    actor_out, rnn_state = drc_net(env_out, rnn_state)

    episode_length = 0
    board_num = 0

    episode_returns = []
    while(len(episode_returns) < num_episodes):
        if episode_length > 0:
            step_entry["reward"] = reward.item()
            episode_entry.append(step_entry)
        print(board_num)
        state, reward, done, info = env.step(actor_out.action)
        env_out = util.create_env_out(actor_out.action, state, reward, done, info, flags)
        actor_out, rnn_state = drc_net(env_out, rnn_state)

        step_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in feature_fncs}
        step_entry["action"] = actor_out.action.item()
        step_entry["board_state"] = state["real_states"][0]
        step_entry["hidden_states"] = drc_net.hidden_state
        step_entry["board_num"] = board_num
        episode_length += 1

        if done:
            for step, step_entry in enumerate(episode_entry):
                step_entry["episode_length"] = episode_length
                step_entry["steps_remaining"] = episode_length - step
                step_entry["action_plus1"] = episode_entry[step+1]["action"] if step < episode_length-2 else 9
                step_entry["action_plus2"] = episode_entry[step+2]["action"] if step < episode_length-3 else 9
                step_entry["action_plus3"] = episode_entry[step+3]["action"] if step < episode_length-4 else 9
                step_entry["action_plus4"] = episode_entry[step+4]["action"] if step < episode_length-5 else 9
                step_entry["action_plus5"] = episode_entry[step+5]["action"] if step < episode_length-6 else 9
                step_entry["reward_plus1"] = episode_entry[step+1]["reward"] if step < episode_length-2 else 9
                step_entry["reward_plus2"] = episode_entry[step+2]["reward"] if step < episode_length-3 else 9
                step_entry["reward_plus3"] = episode_entry[step+3]["reward"] if step < episode_length-4 else 9
                step_entry["reward_plus4"] = episode_entry[step+4]["reward"] if step < episode_length-5 else 9
                step_entry["reward_plus5"] = episode_entry[step+5]["reward"] if step < episode_length-6 else 9
            probing_data += episode_entry
            episode_returns.extend(info["episode_return"][done].tolist())
            episode_length = 0
            board_num += 1

    final_train_board = int(board_num*pct_train)
    probing_train_data = [entry for entry in probing_data if entry["board_num"] <= final_train_board]
    probing_val_data = [entry for entry in probing_data if entry["board_num"] < final_train_board]
    torch.save(ProbingDataset(probing_train_data), "../../probing/data/train_data.pt")
    torch.save(ProbingDataset(probing_val_data), "../../probing/data/val_data.pt")
