import torch
import torch.nn as nn
import torch.nn.functional as F


class ClippedReLU(nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.relu(x), self.min_val, self.max_val)