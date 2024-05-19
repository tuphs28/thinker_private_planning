import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import Callable, Optional
import wandb

from probe_model import DRCProbe
from create_probe_dataset import ProbingDataset

def train_one_epoch(probe: DRCProbe, feature: str, optimiser: torch.optim.Optimizer, loss_fnc: Callable, train_loader: DataLoader, device: torch.device = torch.device("cpu")) -> int:
    train_loss = []
    for transition in train_loader:
        hidden_states = transition["hidden_states"].to(device)
        targets = transition[feature].to(device)
        optimiser.zero_grad()
        probe_logits = probe(hidden_states)
        loss = loss_fnc(probe_logits, targets)
        train_loss.append(loss.item())
        loss.backward()
        optimiser.step()
    return sum(train_loss) / len(train_loss)

@torch.no_grad()
def calc_loss(probe: DRCProbe, feature: str, loss_fnc: Callable, data_loader: DataLoader, device: torch.device = torch.device("cpu")) -> float:
    losses = []
    for transition in data_loader:
        hidden_states = transition["hidden_states"].to(device)
        targets = transition[feature].to(device)
        probe_logits = probe(hidden_states)
        loss = loss_fnc(probe_logits, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)

@torch.no_grad()
def calc_acc(probe: DRCProbe, feature: str, data_loader: DataLoader, device: torch.device = torch.device("cpu")) -> float:
    num_correct = 0
    for transition in data_loader:
        hidden_states = transition["hidden_states"].to(device)
        targets = transition[feature].to(device)
        probe_logits = probe(hidden_states)
        num_correct += torch.sum(probe_logits.argmax(dim=-1)==targets).item()
    return num_correct / len(data_loader.dataset)

def train_probe(probe: DRCProbe, feature: str, n_epochs: int, optimiser: torch.optim.Optimizer, loss_fnc: Callable, train_loader: DataLoader, val_loader: DataLoader, display_loss_freq: int = 1, device: torch.device = torch.device("cpu"), wandb_run: bool = False) -> int:
    train_losses, val_losses = [], []
    for epoch in range(1, n_epochs+1):
        train_loss = train_one_epoch(probe=probe, feature=feature, optimiser=optimiser, loss_fnc=loss_fnc, train_loader=train_loader, device=device)
        train_acc = calc_acc(probe=probe, feature=feature, data_loader=train_loader, device=device)
        val_loss = calc_loss(probe=probe, feature=feature, loss_fnc=loss_fnc, data_loader=val_loader, device=device)
        val_acc = calc_acc(probe=probe, feature=feature, data_loader=val_loader, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if display_loss_freq and epoch % display_loss_freq == 0:
            print(f"EPOCH {epoch} --- Train loss: {train_loss}, Train_acc: {train_acc}, Val loss: {val_loss}, Val acc: {val_acc}")
        if wandb_run:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}) 
    train_output = {"probe": probe, "train_loss": train_losses, "val_loss": val_losses}
    return train_output

def make_trained_probe_for_discrete_feature(probe_args: dict, train_dataset: ProbingDataset, val_dataset: ProbingDataset, test_dataset:ProbingDataset, display_loss_freq: int = 5, wandb_run: bool = False) -> dict:
    assert probe_args["layer"] in [0,1,2], "Please enter a valid DRC layer: [0,1,2]"
    assert probe_args["tick"] in [0,1,2,3], "Please enter a valid DRC tick: [0,1,2,3]"

    min_feature, max_feature = train_dataset.get_feature_range(feature=probe_args["feature"])
    if min_feature != 0:
        for entry in train_dataset.data:
            entry[probe_args["feature"]] -= min_feature
        for entry in val_dataset.data:
            entry[probe_args["feature"]] -= min_feature
        for entry in test_dataset.data:
            entry[probe_args["feature"]] -= min_feature

    probe = DRCProbe(layer=probe_args["layer"],
                     tick=probe_args["tick"],
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
    train_output = train_probe(probe=probe, feature=probe_args["feature"], n_epochs=probe_args["n_epochs"], optimiser=optimiser, loss_fnc=loss_fnc, train_loader=train_loader, val_loader=val_loader, display_loss_freq=display_loss_freq, device=device, wandb_run=wandb_run)
    print(calc_acc(probe=probe, feature=probe_args["feature"], data_loader=test_loader))
    return train_output

if __name__ == "__main__":
    wandb_projname = ""
    probe_args = {
        "feature": "action",
        "layer": 2,
        "tick": 3,
        "target_dim": 5,
        "linear": False,
        "num_layers": 2,
        "hidden_dim": 256,
        "batch_size": 2,
        "optimiser": "Adam",
        "n_epochs": 15,
        "weight_decay": 0.1,
        "lr": 1e-3
    }

    train_dataset = torch.load("./data/train_data.pt")
    val_dataset = torch.load("./data/val_data.pt")
    test_dataset = torch.load("./data/test_data.pt")

    if wandb_projname != "":
        probe_name = "linear" if probe_args["linear"] else f"nonlinear_{probe_args['num_layers']}_{probe_args['hidden_dim']}"
        wandb_expname = f"{probe_args['feature']}_layer_{probe_args['layer']}_tick_{probe_args['tick']}"
        with wandb.init(project=wandb_projname, name=wandb_expname, config=probe_args):
            train_output = make_trained_probe_for_discrete_feature(probe_args=probe_args,
                                                                train_dataset=train_dataset,
                                                                val_dataset=val_dataset,
                                                                test_dataset=test_dataset,
                                                                display_loss_freq=1,
                                                                wandb_run=True)
    else:
        train_output = make_trained_probe_for_discrete_feature(probe_args=probe_args,
                                                            train_dataset=train_dataset,
                                                            val_dataset=val_dataset,
                                                            test_dataset=test_dataset,
                                                            display_loss_freq=1, 
                                                            wandb_run=False)

                                                                                                            
