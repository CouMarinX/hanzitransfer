"""Evaluation metrics for fusion."""

from __future__ import annotations

from typing import Tuple

import torch

from . import ids
from .losses import chamfer_distance


def containment_at(pred: torch.Tensor, base: torch.Tensor, tau: float = 0.5) -> float:
    """Fraction of base pixels covered by ``pred`` above threshold ``tau``."""

    base_mask = base > 0.5
    pred_mask = pred > tau
    if base_mask.sum() == 0:  # pragma: no cover - degenerate
        return 0.0
    return (pred_mask & base_mask).float().mean().item()


def chamfer_dt(pred: torch.Tensor, base: torch.Tensor) -> float:
    return float(chamfer_distance(base, pred))


def ids_validity(pred: torch.Tensor, layout: str) -> float:
    img = pred.squeeze().detach().cpu().numpy()
    guess = ids.guess_layout(1 - img)
    return 1.0 if guess == layout else 0.0
