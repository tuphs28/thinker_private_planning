import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from typing import Callable

from probe_model import LinearProbe
from create_probe_dataset import ProbingDataset

def train_one_epoch(probe: LinearProbe, feature: str, optimiser: torch.optim.Optimizer, loss_fnc: Callable, train_loader: DataLoader) -> int:
    train_loss = []
    for transition in train_loader:
        hidden_states = transition["hidden_states"]
        targets = transition[feature]
        optimiser.zero_grad()
        probe_logits = probe(hidden_states)
        loss = loss_fnc(probe_logits, targets)
        train_loss.append(loss.item())
        loss.backward()
        optimiser.step()
    return sum(train_loss) / len(train_loss)

@torch.no_grad
def calc_loss(probe: LinearProbe, feature: str, loss_fnc: Callable, data_loader: DataLoader) -> float:
    losses = []
    for transition in data_loader:
        hidden_states = transition["hidden_states"]
        targets = transition[feature]
        probe_logits = probe(hidden_states)
        loss = loss_fnc(probe_logits, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)

@torch.no_grad
def calc_acc(probe: LinearProbe, feature: str, data_loader: DataLoader) -> float:
    num_correct = 0
    for transition in data_loader:
        hidden_states = transition["hidden_states"]
        targets = transition[feature]
        probe_logits = probe(hidden_states)
        num_correct += torch.sum(probe_logits.argmax(dim=-1)==targets).item()
    return num_correct / len(data_loader.dataset)

def train_probe(probe: LinearProbe, feature: str, n_epochs: int, optimiser: torch.optim.Optimizer, loss_fnc: Callable, train_loader: DataLoader, val_loader: DataLoader, display_loss_freq: int = 1) -> int:
    train_losses, val_losses = [], []
    for epoch in range(1, n_epochs+1):
        train_loss = train_one_epoch(probe=probe, feature=feature, optimiser=optimiser, loss_fnc=loss_fnc, train_loader=train_loader)
        with torch.no_grad():
            val_loss = calc_loss(probe=probe, feature=feature, loss_fnc=loss_fnc, data_loader=train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if display_loss_freq and epoch % display_loss_freq == 0:
            print(f"EPOCH {epoch} --- Train loss: {train_loss}, Val loss: {val_loss}") 
    train_output = {"probe": probe, "train_loss": train_losses, "val_loss": val_losses}
    return train_output

def make_trained_probe_for_discrete_feature(feature: str, layer: int, tick: int, train_dataset: ProbingDataset, val_dataset: ProbingDataset, test_dataset:ProbingDataset, batch_size: int = 16, n_epochs: int = 20, lr: float = 1e-3, weight_decay: float =  1, optimiser_name: str = "SGD", display_loss_freq: int = 5) -> dict:
    assert layer in [0,1,2], "Please enter a valid DRC layer: [0,1,2]"
    assert tick in [0,1,2,3], "Please enter a valid DRC tick: [0,1,2,3]"
    assert feature in train_dataset[0].keys(), f"Please enter a concept contained in the dataset: {next(iter(train_loader))[0].keys()}"

    min_feature, max_feature = train_dataset.get_feature_range(feature=feature)
    if min_feature != 0:
        for entry in train_dataset.data:
            entry[feature] -= min_feature
        for entry in val_dataset.data:
            entry[feature] -= min_feature
        for entry in test_dataset.data:
            entry[feature] -= min_feature

    probe = LinearProbe(layer=2, tick=3, target_dim=max_feature+1-min_feature)
    loss_fnc = torch.nn.CrossEntropyLoss()
    if optimiser_name == "SGD":
        optimiser = torch.optim.SGD(params=probe.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimiser_name == "Adam":
        optimiser = torch.optim.Adam(params=probe.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Please select a supported optimiser: SGD, Adam")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    train_output = train_probe(probe=probe, feature=feature, n_epochs=n_epochs, optimiser=optimiser, loss_fnc=loss_fnc, train_loader=train_loader, val_loader=val_loader, display_loss_freq=display_loss_freq)
    print(calc_acc(probe=probe, feature=feature, data_loader=val_loader))
    return train_output

if __name__ == "__main__":
    probe_feature = "action"

    train_dataset = torch.load("./data/train_data.pt")
    val_dataset = torch.load("./data/val_data.pt")
    test_dataset = torch.load("./data/test_data.pt")

    train_output = make_trained_probe_for_discrete_feature(feature=probe_feature,
                                                       layer=0,
                                                       tick=0,
                                                       train_dataset=train_dataset,
                                                       val_dataset=val_dataset,
                                                       test_dataset=test_dataset,
                                                       batch_size=4,
                                                       display_loss_freq=1, 
                                                       weight_decay=0,
                                                       n_epochs=30)
                                                    
