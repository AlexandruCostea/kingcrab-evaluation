import argparse
import chess
import time
import torch

from data_sets import CNNDataset
from models import ChessResNetTeacher, DepthwiseCNN
from utils import board_to_cnn_input


def parse_args():
    parser = argparse.ArgumentParser(description='Chess Evaluator Model Training')
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--inferences", type=int, default=10000, help="Number of inferences to run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = 'cpu'
    if "cnn" in args.checkpoint:
        model = DepthwiseCNN()
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
    else:
        raise ValueError("Invalid model checkpoint")

    pos1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    pos2 = "rnbqkbnr/pppppppp/8/8/5N2/8/PPPPPPPP/RNBQKB1R b KQkq - 1 1"

    board1 = chess.Board(pos1)
    board2 = chess.Board(pos2)

    cnn_input1 = board_to_cnn_input(board1).unsqueeze(0)
    cnn_input2 = board_to_cnn_input(board2).unsqueeze(0)
    cnn_input1 = cnn_input1.to(device)
    cnn_input2 = cnn_input2.to(device)



    start_time = time.time()
    for _ in range(args.inferences):
        model.forward(cnn_input1)
        model.forward(cnn_input2)
    
    full_time = time.time() - start_time
    print(f"Time for {args.inferences} inferences: {full_time:.2f} seconds")