"""Evaluation metrics for fusion."""

from __future__ import annotations

from typing import List

import torch

from . import ids
from .ids_grammar import IDSTree, score_ids_legality
from .losses import chamfer_distance, layout_regularizer
from .vector_geom import Path, paths_self_intersections


def containment_at(pred: torch.Tensor, base: torch.Tensor, tau: float = 0.5) -> float:
    """Fraction of base pixels covered by ``pred`` above threshold ``tau``."""

    if base.shape[-2:] != pred.shape[-2:]:
        base = torch.nn.functional.interpolate(
            base.float(), size=pred.shape[-2:], mode="nearest"
        )
    base_mask = base > 0.5
    pred_mask = pred > tau
    if base_mask.sum() == 0:  # pragma: no cover - degenerate
        return 0.0
    return (pred_mask & base_mask).float().mean().item()


def chamfer_dt(pred: torch.Tensor, base: torch.Tensor) -> float:
    if base.shape[-2:] != pred.shape[-2:]:
        base = torch.nn.functional.interpolate(
            base.float(), size=pred.shape[-2:], mode="nearest"
        )
    return float(chamfer_distance(base, pred))


def ids_validity(pred: torch.Tensor, layout: str) -> float:
    img = pred.squeeze().detach().cpu().numpy()
    guess = ids.guess_layout(1 - img)
    return 1.0 if guess == layout else 0.0


def containment_auc(pred: torch.Tensor, base: torch.Tensor, steps: int = 10) -> float:
    """Approximate containment AUC over ``tau`` in ``[0,1]``."""

    taus = torch.linspace(0.0, 1.0, steps)
    vals = [containment_at(pred, base, float(t)) for t in taus]
    return float(torch.tensor(vals).mean().item())


def self_intersection_metric(paths: List[Path]) -> float:
    """Return the number of self intersections for ``paths``."""

    return float(paths_self_intersections(paths))


def layout_score(paths: List[Path], layout: str) -> float:
    """Convert the layout regularizer into a bounded score."""

    penalty = layout_regularizer(paths, layout).item()
    return max(0.0, 1.0 - penalty)


def ids_legality(tree: IDSTree) -> float:
    """Wrapper around :func:`score_ids_legality` for metrics."""

    return score_ids_legality(tree)
