import torch
from torch.utils.data.dataset import Dataset
from create_probe_dataset import ProbingDataset
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import argparse
from train_conv_probe import ConvProbe


parser = argparse.ArgumentParser(description="run f1 exps")
parser.add_argument("--feature", type=str, default="agent_loc_future_trajectory_120")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_name", type=str, default="250m")
args = parser.parse_args()


feature = args.feature
model_name = args.model_name
seed = args.seed
data = torch.load(f"./data/test_data_full_{model_name}.pt")
corrs = []
all_fones = {0: [], 1: [], 2: []}
dprobe2 = ConvProbe(32,5, 1, 0)
dprobe2.load_state_dict(torch.load(f"./convresults/models/{feature}/{model_name}_layer2_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
dprobe1 = ConvProbe(32,5, 1, 0)
dprobe1.load_state_dict(torch.load(f"./convresults/models/{feature}/{model_name}_layer1_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
dprobe0 = ConvProbe(32,5, 1, 0)
dprobe0.load_state_dict(torch.load(f"./convresults/models/{feature}/{model_name}_layer0_kernel1_wd0.001_seed{seed}.pt", map_location=torch.device('cpu')))
probes = [dprobe0, dprobe1, dprobe2]
for step in range(1,20):
    all_preds0, all_preds1, all_preds2, all_tars = [[], [], []], [[], [], []], [[], [], []], []
    for k in range(len(data)):
        trans = data[k]
        for layer, layer_idx in [(0, 32), (1, 96), (2, 160)]:
            if trans["steps_taken"] == step:
                logits0, _ = probes[layer](trans["hidden_states"][1,layer_idx:layer_idx+32,:,:])
                logits1, _ = probes[layer](trans["hidden_states"][2,layer_idx:layer_idx+32,:,:])
                logits2, _ = probes[layer](trans["hidden_states"][3,layer_idx:layer_idx+32,:,:])

                preds0, preds1, preds2 = logits0.argmax(dim=0).view(-1), logits1.argmax(dim=0).view(-1), logits2.argmax(dim=0).view(-1)
                all_preds0[layer] += logits0.argmax(dim=0).view(-1).tolist()
                all_preds1[layer] += logits1.argmax(dim=0).view(-1).tolist()
                all_preds2[layer] += logits2.argmax(dim=0).view(-1).tolist()
                if layer == 0:
                    all_tars += trans[feature].view(-1).tolist()
    for layer in [0, 1, 2]:
        precisions0, recalls0, fones0, _ = precision_recall_fscore_support(all_tars, all_preds0[layer], average=None, zero_division=1, labels=[0,1,2,3,4])
        precisions1, recalls1, fones1, _ = precision_recall_fscore_support(all_tars, all_preds1[layer], average=None, zero_division=1, labels=[0,1,2,3,4])
        precisions2, recalls2, fones2, _ = precision_recall_fscore_support(all_tars, all_preds2[layer], average=None, zero_division=1, labels=[0,1,2,3,4])
        all_fones[layer]+= [fones0[1:].mean(), fones1[1:].mean(), fones2[1:].mean()]
    print(all_fones)
pd.DataFrame(all_fones).to_csv(f"./fone_results/{model_name}_{feature}_{seed}.csv")
