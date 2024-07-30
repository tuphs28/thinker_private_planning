import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from create_probe_dataset import ProbingDataset, ProbingDatasetCleaned
from typing import Optional

class ConvProbe(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()
    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        out = self.conv(input)
        if targets is not None:
            assert out.shape[0] == targets.shape[0]
            out = out.view(out.shape[0], self.out_dim, 64)
            targets = targets.view(out.shape[0], 64)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
    
if __name__ == "__main__":
    import pandas as pd

    channels = list(range(32))
    batch_size = 16
    num_epochs = 400
    out_dim = 5
    probe_args = {}

    features = [f"agent_onto_after"]
    #features = ["tar_next_current_0"]
    layers = [("layer0", 32), ("layer1", 96), ("layer2", 160)]
    #layers = [("layer2", 160)]
    for feature in features:
        print(f"=================================== FEATURE: {feature} =========================================")

        probe_args["feature"] = feature        
        probe_args["positive_feature"] = feature
        results = {}

        train_dataset = torch.load("./data/train_data_random.pt")
        test_dataset = torch.load("./data/test_data_full.pt")
        val_dataset = torch.load("./data/val_data_full.pt")
        cleaned_train_data, cleaned_test_data, cleaned_val_data = [], [], []
        for trans in train_dataset.data:
            if type(trans[probe_args["feature"]]) == int:
                if trans[probe_args["feature"]] != -1:
                    cleaned_train_data.append(trans)
            else:
                cleaned_train_data.append(trans)
        for trans in test_dataset.data:
            if type(trans[probe_args["feature"]]) == int:
                if trans[probe_args["feature"]] != -1:
                    cleaned_test_data.append(trans)
            else:
                cleaned_test_data.append(trans)
        for trans in val_dataset.data:
            if type(trans[probe_args["feature"]]) == int:
                if trans[probe_args["feature"]] != -1:
                    cleaned_val_data.append(trans)
            else:
                cleaned_val_data.append(trans)
        train_dataset.data = cleaned_train_data
        test_dataset.data = cleaned_test_data
        val_dataset.data = cleaned_val_data

        cleaned_train_data = [(trans["hidden_states"], trans[probe_args["feature"]], trans[probe_args["positive_feature"]] if probe_args["positive_feature"] is not None else torch.zeros_like(trans[probe_args["feature"]])) for trans in train_dataset.data]
        cleaned_val_data = [(trans["hidden_states"], trans[probe_args["feature"]], trans[probe_args["positive_feature"]] if probe_args["positive_feature"] is not None else torch.zeros_like(trans[probe_args["feature"]])) for trans in val_dataset.data]
        cleaned_test_data = [(trans["hidden_states"], trans[probe_args["feature"]], trans[probe_args["positive_feature"]] if probe_args["positive_feature"] is not None else torch.zeros_like(trans[probe_args["feature"]])) for trans in test_dataset.data]
        train_dataset = ProbingDatasetCleaned(cleaned_train_data)
        val_dataset = ProbingDatasetCleaned(cleaned_val_data)
        test_dataset = ProbingDatasetCleaned(cleaned_test_data)

        for layer_name, layer_idx in layers:
            print(f"========= {layer_name} =========")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            convprobe = ConvProbe(in_channels=len(channels), out_dim=out_dim, kernel_size=1, padding=0)
            optimiser = torch.optim.Adam(params=convprobe.parameters(), lr=1e-4)

            for epoch in range(1, num_epochs+1):

                positive_accs = [0 for _ in range(out_dim)]
                prop_pos_cors = [0 for _ in range(out_dim)]
                precisions = [0 for _ in range(out_dim)]
                recalls = [0 for _ in range(out_dim)]
                fones = [0 for _ in range(out_dim)]
                conf_mat = [[0 for i in range(out_dim)] for j in range(out_dim)]
                num_true = [0 for i in range(out_dim)]
                num_preds = [0 for i in range(out_dim)]

                for hiddens, targets, _ in train_loader:
                    hiddens = hiddens[:,-1,[layer_idx+c for c in channels],:,:]
                    targets = targets.to(torch.long)
                    optimiser.zero_grad()
                    logits, loss = convprobe(hiddens, targets)
                    loss.backward()
                    optimiser.step()
                full_acc = 0
                positive_acc = 0
                prop_pos_cor = 0
                if epoch % 25 == 0:
                    with torch.no_grad():
                        for hiddens, targets, positive_targets in val_loader:
                            hiddens = hiddens[:,-1,[layer_idx+c for c in channels],:,:]
                            targets = targets.to(torch.long)
                            logits, loss = convprobe(hiddens, targets)
                            full_acc += (torch.sum(logits.argmax(dim=1)==targets.view(-1,64)).item())
                            if probe_args["positive_feature"] is not None:
                                for i in range(positive_targets.shape[0]):
                                    for j in range(out_dim):
                                        positive_accs[j] += 1 if (positive_targets[i]==j).sum().item()==0 else (torch.sum((logits[[i],:,:].argmax(dim=1)==targets[[i],:,:].view(-1,64))[positive_targets[[i],:,:].view(-1,64)==j]).item()) / (positive_targets[i]==j).sum().item()
                                        prop_pos_cors[j] += 1 if (logits[i,:,:].argmax(dim=0)==j).sum().item()==0 else (torch.sum((logits[[i],:,:].argmax(dim=1)==targets[[i],:,:].view(-1,64))[logits[[i],:,:].argmax(dim=1)==j]).item()) / (logits[i,:,:].argmax(dim=0)==j).sum().item()
                                        for k in range(out_dim):
                                            conf_mat[j][k] += torch.sum((logits[[i],:,:].argmax(dim=1)==k)[positive_targets[[i],:,:].view(-1,64)==j]).item()
                                        num_true[j] += torch.sum(positive_targets[[i],:,:].view(-1,64)==j).item()
                                        num_preds[j] += torch.sum((logits[[i],:,:].argmax(dim=1)==j)).item()
                                        #print(conf_mat)
                                        #print(num_preds)

                        print(f"---- Epoch {epoch} -----")
                        print("Full acc:", full_acc/(len(val_dataset.data)*64))
                        if probe_args["positive_feature"] is not None and out_dim == 1:
                            precision = prop_pos_cor / (len(val_dataset.data))
                            recall = positive_acc / (len(val_dataset.data))
                            fone = 0 if precision+recall==0 else (2*precision*recall) / (precision + recall)
                            print("Recall: ", recall)
                            print("Precision: ", precision)
                            print("F1: ", fone)
                        elif probe_args["positive_feature"] is not None:
                            for j in range(out_dim):
                                print(f"-- Out Dim {j} --")
                                precisions[j] = conf_mat[j][j] / sum(conf_mat[j])
                                recalls[j] = conf_mat[j][j] / sum([conf_mat[k][j] for k in range(out_dim)])
                                fones[j] = 0 if precisions[j]+recalls[j]==0 else (2*precisions[j]*recalls[j]) / (precisions[j] + recalls[j])
                                print("Recall:", precisions[j])
                                print("Precision:", recalls[j])
                                print("F1: ", fones[j])

            if out_dim != 1:
                results_dict = {"Acc": full_acc/(len(val_dataset.data)*64)}
                for j in range(out_dim):
                    results_dict[f"Precision_{j}"] = precisions[j]
                    results_dict[f"Recall_{j}"] = recalls[j]
                    results_dict[f"F1_{j}"] = fones[j]
                    results_dict[f"Weights_{j}"] = convprobe.conv.weight.view(out_dim,len(channels))[j,:].tolist()
                results[layer_name] = results_dict

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"./convresults/{feature}.csv")