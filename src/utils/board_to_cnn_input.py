import chess
from typing import List

import torch


piece_channels = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

def board_to_cnn_input(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        tensor[piece_channels[piece.symbol()], rank, file] = 1.0

    return tensor

