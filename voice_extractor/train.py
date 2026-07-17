"""Train a noise-type classifier on the pre-extracted UrbanSound8K features.

Run from a clone of the repository (the feature CSVs live in data/features):
    python -m voice_extractor.train --arch unet
    python -m voice_extractor.train --arch cnn
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models import ARCHITECTURES, ASSETS_DIR, FEATURE_SHAPE, SEED, save_classifier


def load_split(data_dir, name):
    x = np.loadtxt(data_dir / f"{name}_data.csv", delimiter=",", dtype="float32")
    y = np.loadtxt(data_dir / f"{name}_labels.csv", delimiter=",").astype("int64")
    x = x.reshape(-1, 1, *FEATURE_SHAPE)
    return torch.from_numpy(x), torch.from_numpy(y)


def accuracy(model, x, y, batch_size=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start : start + batch_size])
            correct += (logits.argmax(dim=1) == y[start : start + batch_size]).sum().item()
    return correct / len(x)


def train(arch, epochs, batch_size, lr, data_dir, seed=SEED):
    torch.manual_seed(seed)
    x_train, y_train = load_split(data_dir, "train")
    x_test, y_test = load_split(data_dir, "test")
    print(f"train: {len(x_train)} samples, test: {len(x_test)} samples")

    model = ARCHITECTURES[arch]()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {total_loss / len(x_train):.4f}")

    train_acc = accuracy(model, x_train, y_train)
    test_acc = accuracy(model, x_test, y_test)
    print(f"final: train accuracy {train_acc:.2%}, test accuracy {test_acc:.2%}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the noise-type classifier.")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="cnn",
                        help="model architecture (default: cnn)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "features",
                        help="directory with the feature CSVs (default: data/features)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not (args.data_dir / "train_data.csv").is_file():
        sys.exit(f"error: feature CSVs not found in {args.data_dir}\n"
                 "Run from a clone of the repository, or pass --data-dir.")

    model = train(args.arch, args.epochs, args.batch_size, args.lr, args.data_dir, args.seed)
    out = ASSETS_DIR / f"denoiser_{args.arch}.pt"
    save_classifier(model, args.arch, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
