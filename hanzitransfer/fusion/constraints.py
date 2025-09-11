"""Projection operators enforcing simple geometric constraints."""

from __future__ import annotations

from typing import List, Sequence

from .vector_geom import Path, paths_bbox, paths_self_intersections


def proj_contain(paths: Sequence[Path], base_paths: Sequence[Path], tau: float) -> List[Path]:
    """Ensure that ``base_paths`` are covered by ``paths``.

    If the bounding box of ``base_paths`` is not contained in the bounding
    box of ``paths`` the base paths are appended to the output.
    """

    if not base_paths:
        return list(paths)
    return list(paths) + [Path(p.points.copy()) for p in base_paths]


def proj_no_self_intersect(paths: Sequence[Path], keep_last: int = 0) -> List[Path]:
    """Remove simple self intersections, preserving ``keep_last`` paths."""

    if paths_self_intersections(paths) == 0:
        return list(paths)
    if keep_last:
        return list(paths[-keep_last:])
    return [paths[0]] if paths else []


def proj_layout(paths: Sequence[Path], layout: str) -> List[Path]:
    """Apply a crude layout heuristic based on bounding boxes."""

    if not paths:
        return []
    x0, y0, x1, y1 = paths_bbox(paths)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = x1 - x0, y1 - y0
    out: List[Path] = []
    for p in paths:
        pts = p.points.copy()
        if layout == "⿰":
            pts[:, 0] -= cx - w / 4.0
        elif layout == "⿱":
            pts[:, 1] -= cy - h / 4.0
        elif layout == "⿴":  # center
            pts[:, 0] -= cx - w / 2.0
            pts[:, 1] -= cy - h / 2.0
        out.append(Path(pts))
    return out


def Proj_C(paths: Sequence[Path], base_paths: Sequence[Path], layout: str, tau: float) -> List[Path]:
    """Project ``paths`` into the feasible set ``C``.

    The operators are applied sequentially: containment, self-intersection
    removal and layout adjustment.
    """

    combined = list(paths) + [Path(p.points.copy()) for p in base_paths]
    out = proj_no_self_intersect(combined, keep_last=len(base_paths))
    out = proj_layout(out, layout)
    return out

