import torch
import chess
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

from utils import board_to_cnn_input



class CNNDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = []
        for entry in data:
            if entry["eval"] > 600 or entry["eval"] < -600:
                continue
   
            self.data.append(entry)


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fen = self.data[idx]["fen"]
        eval_cp = self.data[idx]["eval"] / 600.0

        board = chess.Board(fen)
        cnn_input = board_to_cnn_input(board)

        return cnn_input, torch.tensor(eval_cp, dtype=torch.float32)