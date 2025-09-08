"""Simple stroke sequence renderer using Pillow."""
from typing import List, Dict, Any
from PIL import Image, ImageDraw


def render_strokes(strokes: List[List[Dict[str, Any]]], size: int = 64, width: int = 2) -> Image.Image:
    """Render a list of strokes into a grayscale image.

    Each stroke is a list of commands produced by ``stroke_dataset.glyph_to_strokes``.
    """
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        current = None
        for cmd in stroke:
            op = cmd["op"]
            pts = cmd["points"]
            if op == "M":
                current = tuple(pts[0])
            elif op == "L" and current is not None:
                next_pt = tuple(pts[0])
                draw.line([current, next_pt], fill=0, width=width)
                current = next_pt
            elif op == "Q" and current is not None:
                # approximate quadratic curve with polyline
                for pt in pts:
                    next_pt = tuple(pt)
                    draw.line([current, next_pt], fill=0, width=width)
                    current = next_pt
            elif op == "C" and current is not None:
                # cubic Bezier; Pillow expects start + control + control + end
                flat = [current] + [tuple(p) for p in pts]
                draw.bezier([coord for point in flat for coord in point], fill=0, width=width)
                current = tuple(pts[-1])
            elif op == "Z":
                current = None
    return img
