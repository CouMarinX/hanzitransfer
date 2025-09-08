import os
import numpy as np
import pytest

from hanzitransfer.models.cnn import build_cnn, train_cnn


def test_build_cnn_compiles():
    model = build_cnn()
    assert isinstance(model.metrics_names, list)


def test_train_cnn(tmp_path):
    # Create small random dataset
    inputs = np.random.rand(8, 64, 64).astype("float32")
    targets = np.random.rand(8, 64, 64).astype("float32")
    np.save(tmp_path / "inputs.npy", inputs)
    np.save(tmp_path / "targets.npy", targets)

    model_path = tmp_path / "model.keras"
    model = train_cnn(
        data_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        model_path=str(model_path),
    )

    assert os.path.exists(model_path)
    assert model.output_shape == (None, 64, 64, 1)
