import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .custom_layers import ClippedReLU


class LayerStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 16)
        self.act1 = ClippedReLU()
        self.fc2 = nn.Linear(16, 32)
        self.act2 = ClippedReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class HalfKAModel(nn.Module):
    def __init__(self, device: torch.device = torch.device('cpu'), checkpoint_path: Optional[str] = None):
        super().__init__()
        self.device = device

        self.embedding_own = nn.EmbeddingBag(45056, 520, mode='sum', sparse=True)
        self.embedding_opp = nn.EmbeddingBag(45056, 520, mode='sum', sparse=True)

        self.avg_linear = nn.Linear(8, 1)
        self.layer_stacks = nn.ModuleList([LayerStack() for _ in range(8)])

        self.cache = None
        self.precomputed_own = None
        self.precomputed_opp = None

        self.to(device)

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(checkpoint)   
            self.eval()
            self._precompute_embeddings()
            self.cache = None


    def _precompute_embeddings(self):
        with torch.no_grad():
            weights = self.embedding_own.weight 
            self.precomputed_own = weights.detach().clone().to(self.device)

            weights = self.embedding_opp.weight
            self.precomputed_opp = weights.detach().clone().to(self.device)


    def forward(self, own_batch: List[torch.Tensor], opp_batch: List[torch.Tensor]) -> torch.Tensor:
        if self.training or self.precomputed_own is None or self.precomputed_opp is None:
            return self._forward_training(own_batch, opp_batch)
        else:
            return self._forward_inference(own_batch[0], opp_batch[0])


    def _forward_training(self, own_batch: List[torch.Tensor], opp_batch: List[torch.Tensor]):
        def flatten_batch(batch: List[torch.Tensor]):
            flat = []
            offsets = []
            offset = 0
            for indices in batch:
                offsets.append(offset)
                flat.append(indices)
                offset += indices.size(0)
            flat = torch.cat(flat).to(self.device)
            offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
            return flat, offsets

        own_flat, own_offsets = flatten_batch(own_batch)
        opp_flat, opp_offsets = flatten_batch(opp_batch)

        own_embed = self.embedding_own(own_flat, own_offsets)
        opp_embed = self.embedding_opp(opp_flat, opp_offsets)

        own_8, own_512 = own_embed[:, :8], own_embed[:, 8:]
        opp_8, opp_512 = opp_embed[:, :8], opp_embed[:, 8:]

        avg_score = self.avg_linear(own_8 - opp_8)
        x_1024 = torch.cat([own_512, opp_512], dim=1)

        piece_counts = torch.tensor(
            [own.size(0) + 1 for own, opp in zip(own_batch, opp_batch)],
            dtype=torch.long, device=self.device
        )

        bucket_indices = ((piece_counts - 1) // 4).clamp(0, 7)

        outputs = []
        for i, bucket in enumerate(bucket_indices):
            out = self.layer_stacks[bucket](x_1024[i].unsqueeze(0)) + avg_score[i]
            outputs.append(out)

        return torch.cat(outputs).squeeze(1)


    def _forward_inference(self, own_indices: torch.Tensor, opp_indices: torch.Tensor):
        new_own_set = set(own_indices.tolist())
        new_opp_set = set(opp_indices.tolist())

        if self.cache is None:
            self.cache = {
                "own_indices": new_own_set,
                "opp_indices": new_opp_set,
                "own_sum": self.precomputed_own[own_indices].sum(dim=0, keepdim=True),
                "opp_sum": self.precomputed_opp[opp_indices].sum(dim=0, keepdim=True)
            }

        else:
            old_own_set = self.cache["own_indices"]
            old_opp_set = self.cache["opp_indices"]

            own_added = list(new_own_set - old_own_set)
            own_removed = list(old_own_set - new_own_set)

            opp_added = list(new_opp_set - old_opp_set)
            opp_removed = list(old_opp_set - new_opp_set)


            if own_added:
                idx = torch.tensor(own_added, device=self.device)
                self.cache["own_sum"] += self.precomputed_own[idx].sum(dim=0, keepdim=True)
            if own_removed:
                idx = torch.tensor(own_removed, device=self.device)
                self.cache["own_sum"] -= self.precomputed_own[idx].sum(dim=0, keepdim=True)
            self.cache["own_indices"] = new_own_set


            if opp_added:
                idx = torch.tensor(opp_added, device=self.device)
                self.cache["opp_sum"] += self.precomputed_opp[idx].sum(dim=0, keepdim=True)
            if opp_removed:
                idx = torch.tensor(opp_removed, device=self.device)
                self.cache["opp_sum"] -= self.precomputed_opp[idx].sum(dim=0, keepdim=True)
            self.cache["opp_indices"] = new_opp_set


        own_sum = self.cache["own_sum"]
        opp_sum = self.cache["opp_sum"]

        own_8, own_512 = own_sum[:, :8], own_sum[:, 8:]
        opp_8, opp_512 = opp_sum[:, :8], opp_sum[:, 8:]

        avg_score = self.avg_linear(own_8 - opp_8)
        x_1024 = torch.cat([own_512, opp_512], dim=1)

        piece_count = own_indices.size(0) + 1
        bucket_index = (piece_count - 1) // 4

        out = self.layer_stacks[bucket_index](x_1024) + avg_score
        return out.squeeze()