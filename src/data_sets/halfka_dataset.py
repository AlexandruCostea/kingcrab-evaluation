import torch
import chess
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

from utils import compute_halfka_indices


def sparse_dual_collate_fn(batch):
    own_batch, opp_batch, scores = zip(*batch)
    return list(own_batch), list(opp_batch), torch.stack(scores)


class HalfKADataset(Dataset):
    def __init__(self, data: List[Dict]):

        self.data = []
        for item in data:
            if item["eval"] > 600 or item["eval"] < -600:
                continue
            
            self.data.append(item)


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        fen = self.data[idx]["fen"]
        eval_cp = self.data[idx]["eval"] / 600.0

        board = chess.Board(fen)
        own_color = board.turn
        opp_color = not own_color

        own_indices = compute_halfka_indices(board, own_color)
        opp_indices = compute_halfka_indices(board, opp_color)

        return (
            torch.tensor(own_indices, dtype=torch.long),
            torch.tensor(opp_indices, dtype=torch.long),
            torch.tensor(eval_cp, dtype=torch.float32)
        )