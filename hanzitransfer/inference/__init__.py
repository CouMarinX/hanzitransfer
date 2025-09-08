"""Model loading and inference helpers for hanzitransfer."""

from __future__ import annotations

import tensorflow as tf

from .renderer import render_strokes


def load_cnn_model(path: str = "hanzi_style_model.keras") -> tf.keras.Model:
    """Load the CNN style transfer model."""
    return tf.keras.models.load_model(path)


def load_cvae_decoder(path: str = "hanzi_cvae_decoder.keras") -> tf.keras.Model:
    """Load the trained CVAE decoder model."""
    return tf.keras.models.load_model(path)

__all__ = ["load_cnn_model", "load_cvae_decoder", "render_strokes"]
