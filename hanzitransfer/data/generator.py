import os
import itertools
from typing import List, Sequence, Tuple

import numpy as np
from hanzi_chaizi import HanziChaizi
from PIL import Image, ImageDraw, ImageFont

# Common radicals and structure maps
COMMON_RADICALS = ["氵", "亻", "口", "木", "女", "心", "手", "扌", "辶", "艹", "日", "月"]
LEFT_RIGHT_RADICALS = {"氵", "亻", "口", "木", "女", "心", "手", "扌", "辶"}
TOP_BOTTOM_RADICALS = {"艹", "日", "月"}

chaizi_tool = HanziChaizi()


def decompose_hanzi(character: str) -> List[str]:
    """Decompose Chinese character into components."""
    try:
        components = chaizi_tool.query(character)
        if components:
            return components[0]
    except Exception:
        pass
    return []


def generate_bitmap(character: str, font_path: str = "simsun.ttc", size: Tuple[int, int] = (64, 64)) -> Image.Image:
    """Generate a bitmap for a character using PIL."""
    try:
        try:
            font = ImageFont.truetype(font_path, 64)
        except Exception:
            font = ImageFont.load_default()
        image = Image.new("1", size, 1)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), character, font=font, fill=0)
        return image
    except Exception:
        return None


def image_to_array(image: Image.Image) -> List[List[int]]:
    if image is None:
        return []
    width, height = image.size
    return [[1 if image.getpixel((x, y)) == 0 else 0 for x in range(width)] for y in range(height)]


def compose_bitmaps(radical_img: Image.Image, base_img: Image.Image, structure: str = "left-right") -> Image.Image:
    """Compose two bitmaps according to structure."""
    if radical_img is None or base_img is None:
        return None
    if structure == "top-bottom":
        radical_resized = radical_img.resize((64, 32))
        base_resized = base_img.resize((64, 32))
        canvas = Image.new("1", (64, 64), 1)
        canvas.paste(radical_resized, (0, 0))
        canvas.paste(base_resized, (0, 32))
    else:
        radical_resized = radical_img.resize((32, 64))
        base_resized = base_img.resize((32, 64))
        canvas = Image.new("1", (64, 64), 1)
        canvas.paste(radical_resized, (0, 0))
        canvas.paste(base_resized, (32, 0))
    return canvas


def save_synth_results(base_arrays: np.ndarray, target_arrays: np.ndarray, output_dir: str = "output") -> None:
    """Save results to compressed npz files."""
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "synth_base.npz")
    target_path = os.path.join(output_dir, "synth_target.npz")
    np.savez_compressed(base_path, base_arrays=base_arrays)
    np.savez_compressed(target_path, target_arrays=target_arrays)


def generate_dataset(
    radicals: Sequence[str] = None,
    chars: Sequence[str] = None,
    limit: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset.

    Args:
        radicals: radicals to combine.
        chars: characters to use as bases.
        limit: optional limit for number of characters processed.
    Returns:
        Tuple of numpy arrays (base, target).
    """
    if radicals is None:
        radicals = COMMON_RADICALS
    if chars is None:
        chars = [chr(i) for i in range(0x4E00, 0x9E00)]
    if limit:
        chars = chars[:limit]

    decomposition_map = {c: decompose_hanzi(c) for c in chars}
    existing_pairs = set()
    for comps in decomposition_map.values():
        for a, b in itertools.permutations(comps, 2):
            existing_pairs.add((a, b))

    base_arrays: List[List[List[int]]] = []
    target_arrays: List[List[List[int]]] = []

    for base_character in chars:
        base_img = generate_bitmap(base_character)
        base_array = image_to_array(base_img)
        for radical in radicals:
            if (base_character, radical) in existing_pairs or (radical, base_character) in existing_pairs:
                continue
            radical_img = generate_bitmap(radical)
            structure = "top-bottom" if radical in TOP_BOTTOM_RADICALS else "left-right"
            composed = compose_bitmaps(radical_img, base_img, structure)
            base_arrays.append(base_array)
            target_arrays.append(image_to_array(composed))

    base_np = np.array(base_arrays, dtype=np.uint8)
    target_np = np.array(target_arrays, dtype=np.uint8)
    return base_np, target_np


__all__ = [
    "COMMON_RADICALS",
    "decompose_hanzi",
    "generate_bitmap",
    "image_to_array",
    "compose_bitmaps",
    "save_synth_results",
    "generate_dataset",
]
