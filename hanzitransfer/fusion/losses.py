"""Loss functions for fusion training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


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
