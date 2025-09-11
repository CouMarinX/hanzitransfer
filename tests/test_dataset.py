import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hanzitransfer.fusion.datasets import PairsDataset


def test_dataset_shapes(tmp_path):
    arr = np.zeros((1, 16, 16), dtype=np.float32)
    np.savez(
        tmp_path / "pair.npz",
        base_char="木",
        target_char="林",
        layout="⿰",
        img_base=arr,
        img_target=arr,
        dt_base=arr,
        dt_target=arr,
    )
    ds = PairsDataset(tmp_path)
    x, y, meta = ds[0]
    assert x.shape == (4, 16, 16)  # 1 dt + 3 layout channels
    assert y.shape == (1, 16, 16)
    assert meta["base_char"] == "木"
