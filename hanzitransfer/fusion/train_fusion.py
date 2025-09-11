"""Training script for the fusion UNet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .datasets import LAYOUTS, PairsDataset
from .losses import contain_loss, edge_sharpness
from .models.cond_unet import CondUNet

try:  # pragma: no cover - tensorboard may be unavailable
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def train(
    data_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_contain: float,
    lambda_edge: float,
    save_dir: str,
    seed: int = 0,
) -> Path:
    torch.manual_seed(seed)
    dataset = PairsDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cpu")
    model = CondUNet(in_channels=1 + len(LAYOUTS)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=save_dir) if SummaryWriter else None
    best_loss = float("inf")
    save_path = Path(save_dir) / "hanzi_fusion_unet.pt"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total = 0.0
        for step, (x, y, _) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            l1 = F.l1_loss(pred, y)
            contain = contain_loss(pred, x[:, :1])
            edge = edge_sharpness(pred)
            loss = l1 + lambda_contain * contain + lambda_edge * edge
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
        avg = total / max(1, len(loader))
        if writer:
            writer.add_scalar("loss", avg, epoch)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
    if writer:
        writer.close()
    return save_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train fusion UNet")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-contain", type=float, default=3.0, dest="lambda_contain")
    parser.add_argument("--lambda-edge", type=float, default=0.1, dest="lambda_edge")
    parser.add_argument("--save_dir", default="output/fusion/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_contain=args.lambda_contain,
        lambda_edge=args.lambda_edge,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
