import torch
import torch.nn as nn
from .halfka import HalfKAModel


class HalfKABucketEvaluator(nn.Module):
    def __init__(self, halfka_model: HalfKAModel, bucket_index: int):
        super().__init__()
        self.model = halfka_model
        self.bucket = self.model.layer_stacks[bucket_index]

    def forward(self, x_1024: torch.Tensor, avg_score: torch.Tensor):
        return (self.bucket(x_1024) + avg_score).squeeze()
