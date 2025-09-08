"""Command line and GUI prediction helpers."""

from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

from .renderer import render_strokes


def load_model(path: str) -> tf.keras.Model:
    """Load a trained model from ``path``."""
    return tf.keras.models.load_model(path)


def _hanzi_to_array(hanzi: str) -> np.ndarray:
    """Convert a Hanzi character to a 64x64 binary array."""
    font = ImageFont.truetype("simsun.ttc", 64)
    image = Image.new("L", (64, 64), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), hanzi, font=font, fill=0)
    array = np.array(image)
    return (array < 128).astype("float32")


def predict_chars(model: tf.keras.Model, chars: str) -> Image.Image:
    """Generate an image for ``chars`` using ``model``.

    The returned image concatenates predictions for all characters.
    """
    hanzi_list = list(chars)
    input_arrays = [_hanzi_to_array(h) for h in hanzi_list]
    input_data = np.array(input_arrays).reshape(-1, 64, 64, 1)
    output_data = model.predict(input_data)

    images: List[Image.Image] = []
    for pred in output_data:
        img: Image.Image
        if isinstance(pred, (list, tuple)):
            img = render_strokes(pred)
        else:
            try:
                strokes = json.loads(pred)
                img = render_strokes(strokes)
            except Exception:
                arr = np.array(pred)
                if arr.ndim == 1:
                    arr = arr.reshape(64, 64)
                elif arr.ndim == 3:
                    arr = arr.squeeze(-1)
                img = Image.fromarray((arr * 255).astype(np.uint8))
        images.append(img)

    width = 64 * len(images)
    total_image = Image.new("L", (width, 64))
    for i, img in enumerate(images):
        total_image.paste(img, (i * 64, 0))
    return total_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Hanzi images")
    parser.add_argument("--input", type=str, required=True, help="Input characters")
    parser.add_argument("--model", type=str, default="hanzi_style_model.keras", help="Model path")
    parser.add_argument("--output", type=str, default="generated.png", help="Output image file")
    args = parser.parse_args()

    model = load_model(args.model)
    img = predict_chars(model, args.input)
    img.save(args.output)
    print(f"Saved generated image to {args.output}")


if __name__ == "__main__":
    main()
