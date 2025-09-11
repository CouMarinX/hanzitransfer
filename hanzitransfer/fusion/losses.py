"""Loss functions for fusion training."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from .vector_geom import Path, paths_bbox, paths_self_intersections


def contain_loss(output: torch.Tensor, base: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Penalize pixels of ``base`` not covered by ``output``.

    Parameters
    ----------
    output: torch.Tensor
        Generated image tensor in ``[0, 1]``.
    base: torch.Tensor
        Binary base character mask in ``[0, 1]``.
    tau: float
        Threshold for considering a pixel drawn.
    """

    return (base * F.relu(tau - output)).mean()


def chamfer_distance(base: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Approximate Chamfer distance between two binary images."""

    b = base.squeeze().detach().cpu().numpy() > 0.5
    o = output.squeeze().detach().cpu().numpy() > 0.5
    dt_b = distance_transform_edt(~b)
    dt_o = distance_transform_edt(~o)
    loss = dt_b[o].mean() + dt_o[b].mean()
    return torch.tensor(loss, device=base.device, dtype=base.dtype)


def edge_sharpness(image: torch.Tensor) -> torch.Tensor:
    """Laplacian based sharpness prior."""

    kernel = (
        torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=image.dtype, device=image.device)
        .view(1, 1, 3, 3)
    )
    lap = F.conv2d(image, kernel, padding=1)
    return lap.abs().mean()


def self_intersection_penalty(paths: List[Path]) -> torch.Tensor:
    """Return a penalty proportional to the number of self intersections."""

    count = paths_self_intersections(paths)
    return torch.tensor(float(count))


def layout_regularizer(paths: List[Path], layout: str) -> torch.Tensor:
    """Simple hinge-style layout regularizer."""

    if len(paths) < 2:
        return torch.tensor(0.0)
    b1 = paths_bbox([paths[0]])
    b2 = paths_bbox([paths[1]])
    if layout == "⿰":
        w1 = b1[2] - b1[0]
        w2 = b2[2] - b2[0]
        ratio = w1 / (w1 + w2 + 1e-8)
    elif layout == "⿱":
        h1 = b1[3] - b1[1]
        h2 = b2[3] - b2[1]
        ratio = h1 / (h1 + h2 + 1e-8)
    else:
        return torch.tensor(0.0)
    penalty = max(0.0, 0.4 - ratio) + max(0.0, ratio - 0.6)
    return torch.tensor(penalty)
