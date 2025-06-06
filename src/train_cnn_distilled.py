import argparse
import os
import logging
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from data_sets import CNNDataset
from models import ChessResNetTeacher, DepthwiseCNN
from utils import get_first_n_entries

def parse_args():
    parser = argparse.ArgumentParser(description='Distillation Trainer')
    parser.add_argument("--input_zst", help="Path to .zst file")
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--experiment_name', type=str, default='distilled_student')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--teacher_ckpt', type=str, required=True)
    return parser.parse_args()

class Trainer:
    def __init__(self, args):
        self.experiment_dir = f'../experiments/{args.experiment_name}'
        self.checkpoints_dir = f'{self.experiment_dir}/checkpoints'
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        log_path = f'{self.experiment_dir}/{args.experiment_name}.log'
        self.logger = logging.getLogger('distillation')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        with open(f'{self.experiment_dir}/params.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.student = DepthwiseCNN().to(self.device)
        if args.checkpoint:
            self.student.load_state_dict(torch.load(args.checkpoint))

        self.teacher = ChessResNetTeacher().to(self.device)
        self.teacher.load_state_dict(torch.load(args.teacher_ckpt))
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        data = get_first_n_entries(args.input_zst, args.count)
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        train_ds = CNNDataset(train_data)
        val_ds = CNNDataset(val_data)
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=1)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.epochs = args.epochs

    def compute_metrics(self, preds, targets):
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        mse = F.mse_loss(preds, targets).item()
        mae = F.l1_loss(preds, targets).item()
        sign_acc = ((preds * targets) > 0).float().mean().item()
        within_100 = ((preds - targets).abs() < 1.0).float().mean().item()
        return mse, mae, sign_acc, within_100

    def train(self):
        for epoch in range(self.epochs):
            self.student.train()
            total_loss = 0
            all_preds, all_targets = [], []
            for cnn_input, _ in tqdm(self.train_loader, desc=f'Train Epoch {epoch+1}'):
                cnn_input = cnn_input.to(self.device)
                with torch.no_grad():
                    teacher_output = self.teacher(cnn_input).squeeze(-1)
                student_output = self.student(cnn_input).squeeze(-1)
                loss = F.mse_loss(student_output, teacher_output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                all_preds.append(student_output)
                all_targets.append(teacher_output)

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            mse, mae, sign_acc, within_100 = self.compute_metrics(all_preds, all_targets)
            msg = f"Train Epoch {epoch+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, SignAcc: {sign_acc:.4f}, ±100cp: {within_100:.4f}"
            self.logger.info(msg)
            print(msg)

            val_loss = self.evaluate(epoch)
            self.scheduler.step(val_loss)
            torch.save(self.student.state_dict(), f"{self.checkpoints_dir}/epoch_{epoch+1}.pth")

    def evaluate(self, epoch):
        self.student.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for cnn_input, _ in tqdm(self.val_loader, desc=f'Eval Epoch {epoch+1}'):
                cnn_input = cnn_input.to(self.device)
                teacher_output = self.teacher(cnn_input).squeeze(-1)
                student_output = self.student(cnn_input).squeeze(-1)
                all_preds.append(student_output)
                all_targets.append(teacher_output)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        mse, mae, sign_acc, within_100 = self.compute_metrics(all_preds, all_targets)
        msg = f"Eval Epoch {epoch+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, SignAcc: {sign_acc:.4f}, ±100cp: {within_100:.4f}"
        self.logger.info(msg)
        print(msg)
        return mse

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
