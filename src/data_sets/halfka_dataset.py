import torch
import chess
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

from utils import compute_white_halfka_indices, compute_black_halfka_indices


def sparse_dual_collate_fn(batch):
    own_batch, opp_batch, scores = zip(*batch)
    return list(own_batch), list(opp_batch), torch.stack(scores)


class HalfKADataset(Dataset):
    def __init__(self, data: List[Dict]):

        self.data = []
        for item in data:
            fen = item["fen"]
            eval_cp = max(min(item["eval"], 1000), -1000) / 1000.0

            board = chess.Board(fen)
            own_color = board.turn
            opp_color = not own_color

            own_indices = (
                compute_white_halfka_indices(board) if own_color == chess.WHITE
                else compute_black_halfka_indices(board)
            )
            opp_indices = (
                compute_white_halfka_indices(board) if opp_color == chess.WHITE
                else compute_black_halfka_indices(board)
            )

            self.data.append((own_indices, opp_indices, eval_cp))


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        own, opp, score = self.data[idx]
        return (
            torch.tensor(own, dtype=torch.long),
            torch.tensor(opp, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32)
        )