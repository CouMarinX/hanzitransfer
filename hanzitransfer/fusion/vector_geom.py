"""Vector geometry utilities for Hanzi paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Path:
    """Simple polyline path."""

    points: np.ndarray  # shape (N, 2)


Skel = np.ndarray  # Alias for skeleton images


def raster_to_skeleton(img: np.ndarray) -> Skel:
    """Morphologically thin ``img`` to a one-pixel wide skeleton."""

    img = (img > 0).astype("uint8") * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype("uint8")


def skeleton_to_paths(skel: Skel) -> List[Path]:
    """Extract polyline paths from a binary skeleton image."""

    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths: List[Path] = []
    for cnt in contours:
        cnt = cnt.squeeze(1)
        if len(cnt) < 2:
            continue
        paths.append(Path(cnt.astype(float)))
    return paths


def _segments(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    pts = path.points
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def ccw(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    if np.allclose(a1, b1) or np.allclose(a1, b2) or np.allclose(a2, b1) or np.allclose(a2, b2):
        return False
    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)


def paths_self_intersections(paths: Sequence[Path]) -> int:
    """Count the number of self intersections across all paths."""

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for p in paths:
        segments.extend(_segments(p))
    count = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if _intersect(*segments[i], *segments[j]):
                count += 1
    return count


def paths_curvature_stats(paths: Sequence[Path]) -> Dict[str, float]:
    """Return simple curvature statistics for ``paths``."""

    curvatures: List[float] = []
    total_length = 0.0
    for p in paths:
        pts = p.points
        if len(pts) < 2:
            continue
        diffs = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        total_length += float(seg_len.sum())
        if len(diffs) >= 2:
            dirs = diffs / (seg_len[:, None] + 1e-8)
            angles = []
            for i in range(len(dirs) - 1):
                v1, v2 = dirs[i], dirs[i + 1]
                ang = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                angles.append(abs(float(ang)))
            curvatures.extend(angles)
    mean_curv = float(np.mean(curvatures)) if curvatures else 0.0
    max_curv = float(np.max(curvatures)) if curvatures else 0.0
    return {"mean_curvature": mean_curv, "max_curvature": max_curv, "length": total_length}


def paths_bbox(paths: Sequence[Path]) -> Tuple[int, int, int, int]:
    """Compute bounding box encompassing all ``paths``."""

    pts = np.concatenate([p.points for p in paths], axis=0)
    x0, y0 = np.min(pts, axis=0)
    x1, y1 = np.max(pts, axis=0)
    return int(x0), int(y0), int(x1), int(y1)


def rasterize_paths(paths: Sequence[Path], H: int, W: int) -> np.ndarray:
    """Rasterise ``paths`` into a ``H``x``W`` image."""

    img: np.ndarray = np.zeros((H, W), dtype=np.uint8)
    for p in paths:
        pts: np.ndarray = p.points.astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, 255, 1)  # type: ignore[call-overload]
    return img.astype(np.float32) / 255.0

