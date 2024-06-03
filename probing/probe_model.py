import torch
import torch.nn as nn
from torch.nn.functional import softmax
from typing import Optional

class DRCProbe(nn.Module):
    """Linear probe for the DRC(3,3) agent"""
    
    def __init__(self, drc_layer: int, drc_tick: int, target_dim: int, linear: bool = True, num_layers: int = 1, hidden_dim: int = 64, bias: bool = True, drc_channels: Optional[list] = None):
        super().__init__()
        assert drc_layer in [0,1,2,3], "Please chose a valid layer: 0, 1 or 2"
        assert drc_tick in [0,1,2,3], "Please enter a valid tick: 0, 1, 2, or 4"
        self.drc_layer = drc_layer
        self.drc_tick = drc_tick
        self.target_dim = target_dim
        self.linear = linear
        self.hidden_dim = hidden_dim

        if drc_channels == "hidden":
            self.drc_channels = list(range(64*self.drc_layer, 64*self.drc_layer+32))
        elif drc_channels == "cell":
            self.drc_channels = list(range(64*self.drc_layer+32, 64*self.drc_layer+64))
        elif drc_channels == "xenc":
            self.drc_channels = list(range(192, 224))
        elif drc_channels == "cellxenc":
            self.drc_channels = list(range(64*self.drc_layer+32, 64*self.drc_layer+64)) + list(range(192, 224))
        elif drc_channels == "hiddenxenc":
            self.drc_channels = list(range(64*self.drc_layer, 64*self.drc_layer+32)) + list(range(192, 224))
        elif drc_channels == "hiddencell":
            self.drc_channels = list(range(64*self.drc_layer, 64*self.drc_layer+64)) 
        elif drc_channels == "all":
            self.drc_channels = list(range(64*self.drc_layer, 64*self.drc_layer+64)) + list(range(192, 224))
        elif type(drc_channels) == list:
            self.drc_channels = [64*self.drc_layer + c for c in drc_channels]
        else:
            raise ValueError("drc_channels must be either list of ints or a string corresponding to a specific component (hidden, cell, xenc, cellxenc, hiddenxenc, hiddencell, all)")
        #print(self.drc_channels)
        self.in_dim = 64 * len(self.drc_channels)

        if self.linear:
            self.network = nn.Linear(in_features=self.in_dim, out_features=self.target_dim, bias=bias)
        else:
            layers = []
            for layer_idx in range(num_layers):
                if layer_idx == 0:
                    layers += [nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim), nn.ReLU()]
                else:
                    layers += [nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim), nn.ReLU()]
            layers += [nn.Linear(in_features=self.hidden_dim, out_features=self.target_dim)]
            self.network = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.tensor) -> torch.tensor:
        probe_inputs = hidden_states[:,self.drc_tick,self.drc_channels,:,:]
        probe_inputs = probe_inputs.view(hidden_states.shape[0],-1)
        probe_logits = self.network(probe_inputs)
        probe_probs = softmax(probe_logits, dim=-1)
        return probe_probs