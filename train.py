"""
Train the Phase 1 tile classifier on preprocessed Sentinel-2 chips.

Usage:
    python train.py --model classifier
"""
print("SCRIPT STARTED", flush=True)
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from models.classifier import TileClassifier


class TileDataset(Dataset):
    def __init__(self, tiles_dir, split_file):
        self.tiles_dir = tiles_dir
        with open(os.path.join(tiles_dir, split_file)) as f:
            self.entries = [line.strip().split(",") for line in f if line.strip()]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rel_path, label = self.entries[idx]
        chip = np.load(os.path.join(self.tiles_dir, rel_path))
        return torch.from_numpy(chip), torch.tensor([float(label)])


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classifier"], default="classifier")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tiles_dir = os.path.join(config.PROCESSED_DATA_DIR, config.ACTIVE_REGION)
    train_ds = TileDataset(tiles_dir, "train.txt")
    val_ds = TileDataset(tiles_dir, "val.txt")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = TileClassifier(in_channels=len(config.BANDS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best model saved to {ckpt_path}")

    print(f"Training done. Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()