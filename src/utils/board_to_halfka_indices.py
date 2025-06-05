import chess
from typing import List

# White perspective (no white king)
white_piece_types = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4,
    'p': 5, 'n': 6, 'b': 7, 'r': 8, 'q': 9, 'k': 10
}

# Black perspective (no black king)
black_piece_types = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4,
    'P': 5, 'N': 6, 'B': 7, 'R': 8, 'Q': 9, 'K': 10
}


def vertical_flip(square: chess.Square) -> chess.Square:
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square)
    return chess.square(file, rank)


def feature_index(piece_idx: int, piece_sq: int, king_sq: int) -> int:
    if piece_sq == king_sq:
        return None
    return piece_idx * 64 * 64 + piece_sq * 64 + king_sq


def compute_white_halfka_indices(board: chess.Board) -> List[int]:
    king_sq = board.king(chess.WHITE)
    if king_sq is None:
        return []

    indices = []
    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE and piece.piece_type == chess.KING:
            continue

        symbol = piece.symbol()
        if symbol not in white_piece_types:
            continue

        piece_idx = white_piece_types[symbol]
        feat_idx = feature_index(piece_idx, square, king_sq)
        if feat_idx is not None:
            indices.append(feat_idx)

    return indices


def compute_black_halfka_indices(board: chess.Board) -> List[int]:
    king_sq = board.king(chess.BLACK)
    if king_sq is None:
        return []

    king_sq = vertical_flip(king_sq)

    indices = []
    for square, piece in board.piece_map().items():
        if piece.color == chess.BLACK and piece.piece_type == chess.KING:
            continue

        symbol = piece.symbol()
        if symbol not in black_piece_types:
            continue

        flipped_sq = vertical_flip(square)
        piece_idx = black_piece_types[symbol]
        feat_idx = feature_index(piece_idx, flipped_sq, king_sq)
        if feat_idx is not None:
            indices.append(feat_idx)

    return indices