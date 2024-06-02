import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import Callable, Optional
import wandb

from probe_model import DRCProbe
from create_probe_dataset import ProbingDataset, ProbingDatasetCleaned

def train_one_epoch(probe: DRCProbe, optimiser: torch.optim.Optimizer, loss_fnc: Callable, train_loader: DataLoader, device: torch.device = torch.device("cpu")) -> int:
    train_loss = []
    for (hidden_states, targets) in train_loader:
        hidden_states = hidden_states.to(device)
        targets = targets.to(device)
        optimiser.zero_grad()
        probe_logits = probe(hidden_states)
        loss = loss_fnc(probe_logits, targets)
        train_loss.append(loss.item())
        loss.backward()
        optimiser.step()
    return sum(train_loss) / len(train_loss)

@torch.no_grad()
def calc_loss(probe: DRCProbe, loss_fnc: Callable, data_loader: DataLoader, device: torch.device = torch.device("cpu")) -> float:
    losses = []
    for (hidden_states, targets) in data_loader:
        hidden_states = hidden_states.to(device)
        targets = targets.to(device)
        probe_logits = probe(hidden_states)
        loss = loss_fnc(probe_logits, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)

@torch.no_grad()
def calc_acc(probe: DRCProbe, data_loader: DataLoader, device: torch.device = torch.device("cpu")) -> float:
    num_correct = 0
    for (hidden_states, targets) in data_loader:
        hidden_states = hidden_states.to(device)
        targets = targets.to(device)
        probe_logits = probe(hidden_states)
        num_correct += torch.sum(probe_logits.argmax(dim=-1)==targets).item()
    return num_correct / len(data_loader.dataset)

def train_probe(probe: DRCProbe, n_epochs: int, optimiser: torch.optim.Optimizer, loss_fnc: Callable, train_loader: DataLoader, val_loader: DataLoader, display_loss_freq: int = 1, device: torch.device = torch.device("cpu"), wandb_run: bool = False) -> int:
    train_losses, val_losses = [], []
    for epoch in range(1, n_epochs+1):
        train_loss = train_one_epoch(probe=probe, optimiser=optimiser, loss_fnc=loss_fnc, train_loader=train_loader, device=device)
        train_acc = calc_acc(probe=probe, data_loader=train_loader, device=device)
        val_loss = calc_loss(probe=probe, loss_fnc=loss_fnc, data_loader=val_loader, device=device)
        val_acc = calc_acc(probe=probe, data_loader=val_loader, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if display_loss_freq and epoch % display_loss_freq == 0:
            print(f"EPOCH {epoch} --- Train loss: {train_loss:.4f}, Train_acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        if wandb_run:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}) 
    train_output = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
    return train_output

def make_trained_probe_for_discrete_feature(probe_args: dict, train_dataset: ProbingDataset, val_dataset: ProbingDataset, test_dataset:ProbingDataset, display_loss_freq: int = 5, wandb_run: bool = False) -> dict:
    assert probe_args["layer"] in [0,1,2], "Please enter a valid DRC layer: [0,1,2]"
    assert probe_args["tick"] in [0,1,2,3], "Please enter a valid DRC tick: [0,1,2,3]"

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

    if type(train_dataset[0][probe_args["feature"]]) == int:
        min_feature, max_feature = train_dataset.get_feature_range(feature=probe_args["feature"])
        if min_feature != 0:
            for trans in train_dataset.data + val_dataset.data + test_dataset.data:
                trans[probe_args["feature"]] -= min_feature
    elif type(train_dataset[0][probe_args["feature"]]) == tuple: # turn tuple features into discrete int features in a kind of hacky way
        tuple_to_int = {}
        for trans in train_dataset.data + val_dataset.data + test_dataset.data:
            if trans[probe_args["feature"]] not in tuple_to_int.keys():
                tuple_to_int[trans[probe_args["feature"]]] = len(tuple_to_int.keys())
            trans[probe_args["feature"]] = tuple_to_int[trans[probe_args["feature"]]]
        min_feature, max_feature = train_dataset.get_feature_range(feature=probe_args["feature"])
    else:
        raise ValueError(f"The feature you are trying to train a probe for is of a type that is not currently supported - for reference, the feature value of the first element of the training set is: {train_dataset[0][probe_args['feature']]}")        

    cleaned_train_data = [(trans["hidden_states"], trans[probe_args["feature"]]) for trans in train_dataset.data]
    cleaned_val_data = [(trans["hidden_states"], trans[probe_args["feature"]]) for trans in val_dataset.data]
    cleaned_test_data = [(trans["hidden_states"], trans[probe_args["feature"]]) for trans in test_dataset.data]
    train_dataset = ProbingDatasetCleaned(cleaned_train_data)
    val_dataset = ProbingDatasetCleaned(cleaned_val_data)
    test_dataset = ProbingDatasetCleaned(cleaned_test_data)

    probe = DRCProbe(drc_layer=probe_args["layer"],
                     drc_tick=probe_args["tick"],
                     drc_channels=probe_args["channels"],
                     linear=probe_args["linear"],
                     num_layers=probe_args["num_layers"],
                     hidden_dim=probe_args["hidden_dim"],
                     target_dim=max_feature+1-min_feature)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    probe.to(device)

    loss_fnc = torch.nn.CrossEntropyLoss()

    if probe_args["optimiser"] == "SGD":
        optimiser = torch.optim.SGD(params=probe.parameters(), lr=probe_args["lr"], weight_decay=probe_args["weight_decay"])
    elif probe_args["optimiser"] == "Adam":
        optimiser = torch.optim.Adam(params=probe.parameters(), lr=probe_args["lr"], weight_decay=probe_args["weight_decay"])
    else:
        raise ValueError("Please select a supported optimiser: SGD, Adam")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=probe_args["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=probe_args["batch_size"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=probe_args["batch_size"])
    train_output = train_probe(probe=probe, n_epochs=probe_args["n_epochs"], optimiser=optimiser, loss_fnc=loss_fnc, train_loader=train_loader, val_loader=val_loader, display_loss_freq=display_loss_freq, device=device, wandb_run=wandb_run)
    test_acc = calc_acc(probe=probe, data_loader=test_loader)
    train_output["test_acc"] = test_acc
    return train_output

def run_probe_experiments(features: list, drc_layers: list, drc_ticks: list, drc_channels: list = [None], linears: list =[False], wandb_projname: Optional[str] = None):

    probe_args = {
        "linear": False,
        "num_layers": 1,
        "hidden_dim": 1024,
        "batch_size": 32,
        "optimiser": "Adam",
        "n_epochs": 140,
        "weight_decay": 0.0,
        "lr": 1e-3,
        "channels": None
    }

    train_dataset = torch.load("./data/train_data.pt")
    val_dataset = torch.load("./data/val_data.pt")
    test_dataset = torch.load("./data/test_data.pt")
    results = {}

    for feature in features:
        probe_args["feature"] = feature
        for drc_layer in drc_layers:
            probe_args["layer"] = drc_layer
            for drc_tick in drc_ticks:
                probe_args["tick"] = drc_tick
                for drc_channel in drc_channels:
                    probe_args["channels"] = drc_channel
                    for linear in linears:
                        probe_args["linear"] = linear
                        train_dataset = torch.load("./data/train_data.pt")
                        val_dataset = torch.load("./data/val_data.pt")
                        test_dataset = torch.load("./data/test_data.pt")
                        expname = f"{probe_args['feature']}_{'linear' if probe_args['linear'] else 'nonlinear'}_layer{probe_args['layer']}_channel{probe_args['channels']}_tick{probe_args['tick']}" 
                        print(f"---{expname}---")
                        if wandb_projname is not None:
                            with wandb.init(project=wandb_projname, name=expname, config=probe_args):
                                train_output = make_trained_probe_for_discrete_feature(probe_args=probe_args,
                                                                                    train_dataset=train_dataset,
                                                                                    val_dataset=val_dataset,
                                                                                    test_dataset=test_dataset,
                                                                                    display_loss_freq=5,
                                                                                    wandb_run=True)
                        else:
                            train_output = make_trained_probe_for_discrete_feature(probe_args=probe_args,
                                                                                train_dataset=train_dataset,
                                                                                val_dataset=val_dataset,
                                                                                test_dataset=test_dataset,
                                                                                display_loss_freq=20, 
                                                                                wandb_run=False)
                        results[expname] = train_output
    return results



if __name__ == "__main__":
    import pandas as pd
    for feature in ["agent_loc"]:
        channels = [[t] for t in range(64,96)]
        results = run_probe_experiments(features=[feature],
                                    drc_layers=[2],
                                    drc_ticks=[3],
                                    drc_channels=channels,
                                    linears=[True])
        results_df = pd.DataFrame(results)
        filename = f"{feature}_{'multi' if channels[0] == 'hidden' else 'indiv'}"
        results_df.to_csv(f"./results/{filename}.csv")
    
                                                                                                            
