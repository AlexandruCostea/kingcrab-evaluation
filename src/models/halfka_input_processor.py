import torch
import torch.nn as nn
from .halfka import HalfKAModel


class HalfKAInputProcessor(nn.Module):
    def __init__(self, halfka_model: HalfKAModel):
        super().__init__()
        self.model = halfka_model
        self.avg_linear = self.model.avg_linear

    def forward(self, own_sum: torch.Tensor, opp_sum: torch.Tensor):
        own_8, own_512 = own_sum[:, :8], own_sum[:, 8:]
        opp_8, opp_512 = opp_sum[:, :8], opp_sum[:, 8:]

        avg_score = self.avg_linear(own_8 - opp_8)
        x_1024 = torch.cat([own_512, opp_512], dim=1)
        return x_1024, avg_score
