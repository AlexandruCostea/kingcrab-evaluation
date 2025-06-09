import chess
from typing import List

piece_types = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4,
    'p': 5, 'n': 6, 'b': 7, 'r': 8, 'q': 9,
    'k': 10, 'K': 10
}


def vertical_flip(square: chess.Square) -> chess.Square:
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square)
    return chess.square(file, rank)


def feature_index(piece_idx: int, piece_sq: int, king_sq: int) -> int:
    if piece_sq == king_sq:
        return None
    return piece_idx * 64 * 32 + piece_sq * 32 + king_sq


def compute_halfka_indices(board: chess.Board, side: chess.Color) -> List[int]:
    king_sq = board.king(side)
    if king_sq is None:
        return []

    flip = chess.square_rank(king_sq) >= 4
    king_sq = vertical_flip(king_sq) if flip else king_sq


    indices = []
    for square, piece in board.piece_map().items():
        if piece.color == side and piece.piece_type == chess.KING:
            continue


        symbol = piece.symbol()
        if symbol not in piece_types:
            continue

        piece_idx = piece_types[symbol]
        piece_sq = vertical_flip(square) if flip else square
        feat_idx = feature_index(piece_idx, piece_sq, king_sq)
        if feat_idx is not None:
            indices.append(feat_idx)

    return indices