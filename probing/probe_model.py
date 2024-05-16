import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.fc = nn.Linear(in_features=input_dim, target_dim=self.target_dim)
    def forward(self, )
