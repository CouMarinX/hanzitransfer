"""Simple raster to SVG vectorization using OpenCV contours."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def vectorize(image: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Return a list of contour paths for the binary ``image``."""

    img = (image > 0.5).astype("uint8") * 255
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths: List[List[Tuple[int, int]]] = []
    for cnt in contours:
        cnt = cnt.squeeze(1)
        paths.append([(int(x), int(y)) for x, y in cnt])
    return paths


def save_svg(paths: Sequence[Sequence[Tuple[int, int]]], size: Tuple[int, int], path: str | Path) -> None:
    """Save contours to a very small SVG file."""

    w, h = size
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        f.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>\n")
        for pts in paths:
            if not pts:
                continue
            d = "M " + " L ".join(f"{x} {y}" for x, y in pts) + " Z"
            f.write(f"<path d='{d}' fill='black'/>\n")
        f.write("</svg>\n")
