import argparse
import chess
import time
import torch

from data_sets import HalfKADataset, sparse_dual_collate_fn
from models import HalfKAModel
from utils import compute_halfka_indices


def parse_args():
    parser = argparse.ArgumentParser(description='Chess Evaluator Model Training')
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--inferences", type=int, default=100000, help="Number of inferences to run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = 'cpu'
    if "halfka" in args.checkpoint:
        model = HalfKAModel(device=device, checkpoint_path=args.checkpoint).to(device)
        model.eval()
    else:
        raise ValueError("Invalid model checkpoint")

    pos1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    pos2 = "rnbqkbnr/pppppppp/8/8/5N2/8/PPPPPPPP/RNBQKB1R b KQkq - 1 1"

    board1 = chess.Board(pos1)
    board2 = chess.Board(pos2)

    own_indices1 = compute_halfka_indices(board1, chess.WHITE)
    opp_indices1 = compute_halfka_indices(board1, chess.BLACK)
    own_indices2 = compute_halfka_indices(board2, chess.BLACK)
    opp_indices2 = compute_halfka_indices(board2, chess.WHITE)

    own_indices1 = [torch.tensor(own_indices1, dtype=torch.long)]
    opp_indices1 = [torch.tensor(opp_indices1, dtype=torch.long)]
    own_indices2 = [torch.tensor(own_indices2, dtype=torch.long)]
    opp_indices2 = [torch.tensor(opp_indices2, dtype=torch.long)]


    start_time = time.time()
    for _ in range(args.inferences):
        model.forward(own_indices1, opp_indices1)
        model.forward(own_indices2, opp_indices2)
    
    full_time = time.time() - start_time
    print(f"Time for {args.inferences} inferences: {full_time:.2f} seconds")