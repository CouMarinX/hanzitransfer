import argparse
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_cnn(learning_rate: float = 0.001) -> tf.keras.Model:
    """Build and compile a simple CNN model.

    The network maps a 64x64 grayscale image to another 64x64 image. The
    architecture is intentionally small so that unit tests can train it on a
    tiny random dataset in a reasonable amount of time.

    Parameters
    ----------
    learning_rate: float, optional
        Learning rate for the Adam optimizer.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(64 * 64, activation="sigmoid"),
        layers.Reshape((64, 64, 1)),
    ])

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])
    return model


def _load_arrays(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load training arrays from ``data_dir``.

    The directory is expected to contain two files: ``inputs.npy`` and
    ``targets.npy``. Both should be arrays of shape ``(N, 64, 64)`` with
    ``float32`` values in the ``[0, 1]`` range. The function reshapes them to
    ``(N, 64, 64, 1)`` for use with the CNN.
    """

    inputs_path = os.path.join(data_dir, "inputs.npy")
    targets_path = os.path.join(data_dir, "targets.npy")
    if not (os.path.exists(inputs_path) and os.path.exists(targets_path)):
        raise FileNotFoundError(
            "inputs.npy and targets.npy must exist in the provided data directory"
        )

    inputs = np.load(inputs_path).astype("float32")[..., None]
    targets = np.load(targets_path).astype("float32")[..., None]
    return inputs, targets


def train_cnn(
    data_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float = 0.001,
    model_path: Optional[str] = None,
) -> tf.keras.Model:
    """Train the CNN model using data from ``data_dir``.

    Parameters
    ----------
    data_dir: str
        Directory containing ``inputs.npy`` and ``targets.npy``.
    epochs: int
        Number of epochs to train for.
    batch_size: int
        Size of each training batch.
    learning_rate: float, optional
        Learning rate for the optimizer.
    model_path: str, optional
        File path to save the trained model. If ``None``, the model is not
        saved.

    Returns
    -------
    tf.keras.Model
        The trained Keras model.
    """

    inputs, targets = _load_arrays(data_dir)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.shuffle(len(inputs)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_cnn(learning_rate=learning_rate)
    model.fit(dataset, epochs=epochs, verbose=1)

    if model_path:
        model.save(model_path)

    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training the CNN."""
    parser = argparse.ArgumentParser(description="Train a simple CNN model")
    parser.add_argument("--data_dir", required=True, help="Directory containing inputs.npy and targets.npy")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument("--model_path", type=str, default="cnn_model.keras", help="Path to save the trained model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cnn(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
    )
