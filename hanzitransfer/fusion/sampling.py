"""Guided and projected sampling utilities."""

from __future__ import annotations

import torch

from . import constraints, vector_geom


def guided_projected_sampling(
    model,
    cond: torch.Tensor,
    base_img: torch.Tensor,
    layout: str,
    steps: int = 1,
    guide_lambda: float = 3.0,
    proj_every: int = 1,
    tau: float = 0.5,
    seed: int = 0,
):
    """Run a tiny two-stage sampler with projection onto ``C``.

    The implementation here is intentionally lightweight and deterministic.
    """

    del guide_lambda  # guidance is not implemented in this minimal version
    torch.manual_seed(seed)
    with torch.no_grad():
        y = model(cond)
    base_skel = vector_geom.raster_to_skeleton(base_img.squeeze().cpu().numpy())
    base_paths = vector_geom.skeleton_to_paths(base_skel)

    img = y
    paths = []
    for step in range(steps):
        skel = vector_geom.raster_to_skeleton(img.squeeze().cpu().numpy())
        paths = vector_geom.skeleton_to_paths(skel)
        if (step + 1) % proj_every == 0:
            paths = constraints.Proj_C(paths, base_paths, layout, tau)
        ras = vector_geom.rasterize_paths(paths, img.shape[-2], img.shape[-1])
        img = torch.from_numpy(ras).to(cond.device).unsqueeze(0)
    return img, paths

