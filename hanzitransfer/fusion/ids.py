"""Minimal IDS utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont as PILImageFont

LAYOUTS = ["⿰", "⿱", "⿴"]


def render_char(ch: str, size: int = 128) -> np.ndarray:
    """Render a character to a grayscale numpy array."""
    font: PILImageFont.ImageFont
    try:
        font = PILImageFont.truetype("NotoSansCJK-Regular.ttc", size)
    except Exception:  # pragma: no cover - font fallback
        try:
            font = PILImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            font = PILImageFont.load_default()
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, font=font, fill=0)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.min() > 0.9:  # failed to render, draw synthetic
        draw.rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=0)
        arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def guess_layout(image: np.ndarray | Image.Image) -> str:
    """Guess a simple layout for a glyph.

    The heuristic uses mass distribution to decide between left-right (⿰),
    top-bottom (⿱) or surround (⿴).
    """

    if isinstance(image, Image.Image):
        arr = np.array(image, dtype=np.float32) / 255.0
    else:
        arr = image
    arr = 1.0 - arr  # strokes
    h, w = arr.shape
    left = arr[:, : w // 2].sum()
    right = arr[:, w // 2 :].sum()
    top = arr[: h // 2, :].sum()
    bottom = arr[h // 2 :, :].sum()
    if abs(left - right) > abs(top - bottom):
        return "⿰"
    if abs(top - bottom) > abs(left - right):
        return "⿱"
    return "⿴"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Guess IDS layout")
    parser.add_argument("--char", default="木", help="Character to inspect")
    parser.add_argument(
        "--out", default="output/fusion/ids_debug", help="Directory for debug image"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    img = render_char(args.char, size=128)
    layout = guess_layout(img)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Image.fromarray((img * 255).astype("uint8")).save(Path(args.out) / f"{args.char}.png")
    print(layout)


if __name__ == "__main__":  # pragma: no cover
    main()
