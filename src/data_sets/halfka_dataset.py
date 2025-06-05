import torch
from torch.utils.data import Dataset
import chess
from typing import List, Dict, Tuple


def sparse_dual_collate_fn(batch):
    own_batch, opp_batch, scores = zip(*batch)
    return list(own_batch), list(opp_batch), torch.stack(scores)


class HalfKADataset(Dataset):
    def __init__(self, data: List[Dict]):
        """
        Args:
            data: List of dicts with keys ["fen", "eval"]
        """

        # White perspective (no white king)
        self.white_piece_types = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4,
            'p': 5, 'n': 6, 'b': 7, 'r': 8, 'q': 9, 'k': 10
        }

        # Black perspective (no black king)
        self.black_piece_types = {
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4,
            'P': 5, 'N': 6, 'B': 7, 'R': 8, 'Q': 9, 'K': 10
        }

        # self.data = data
        self.data = []
        for item in data:
            fen = item["fen"]
            eval_cp = max(min(item["eval"], 1000), -1000) / 1000.0

            board = chess.Board(fen)
            own_color = board.turn
            opp_color = not own_color

            own_indices = (
                self.compute_white_halfka_indices(board) if own_color == chess.WHITE
                else self.compute_black_halfka_indices(board)
            )
            opp_indices = (
                self.compute_white_halfka_indices(board) if opp_color == chess.WHITE
                else self.compute_black_halfka_indices(board)
            )

            self.data.append((own_indices, opp_indices, eval_cp))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # fen = self.data[idx]["fen"]
        # eval_cp = self.data[idx]["eval"]
        # board = chess.Board(fen)
        # own_color = board.turn
        # opp_color = not own_color

        # own_indices = (
        #     self.compute_white_halfka_indices(board) if own_color == chess.WHITE
        #     else self.compute_black_halfka_indices(board)
        # )
        # opp_indices = (
        #     self.compute_white_halfka_indices(board) if opp_color == chess.WHITE
        #     else self.compute_black_halfka_indices(board)
        # )

        # return (
        #     torch.tensor(own_indices, dtype=torch.long),
        #     torch.tensor(opp_indices, dtype=torch.long),
        #     torch.tensor(eval_cp, dtype=torch.float32)
        # )

        own, opp, score = self.data[idx]
        return (
            torch.tensor(own, dtype=torch.long),
            torch.tensor(opp, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32)
        )

    def vertical_flip(self, square: chess.Square) -> chess.Square:
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        return chess.square(file, rank)

    def feature_index(self, piece_idx: int, piece_sq: int, king_sq: int) -> int:
        if piece_sq == king_sq:
            return None
        return piece_idx * 64 * 64 + piece_sq * 64 + king_sq

    def compute_white_halfka_indices(self, board: chess.Board) -> List[int]:
        king_sq = board.king(chess.WHITE)
        if king_sq is None:
            return []

        indices = []
        for square, piece in board.piece_map().items():
            if piece.color == chess.WHITE and piece.piece_type == chess.KING:
                continue

            symbol = piece.symbol()
            if symbol not in self.white_piece_types:
                continue

            piece_idx = self.white_piece_types[symbol]
            feat_idx = self.feature_index(piece_idx, square, king_sq)
            if feat_idx is not None:
                indices.append(feat_idx)

        return indices

    def compute_black_halfka_indices(self, board: chess.Board) -> List[int]:
        king_sq = board.king(chess.BLACK)
        if king_sq is None:
            return []

        king_sq = self.vertical_flip(king_sq)

        indices = []
        for square, piece in board.piece_map().items():
            if piece.color == chess.BLACK and piece.piece_type == chess.KING:
                continue

            symbol = piece.symbol()
            if symbol not in self.black_piece_types:
                continue

            flipped_sq = self.vertical_flip(square)
            piece_idx = self.black_piece_types[symbol]
            feat_idx = self.feature_index(piece_idx, flipped_sq, king_sq)
            if feat_idx is not None:
                indices.append(feat_idx)

        return indices
