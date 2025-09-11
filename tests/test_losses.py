import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from hanzitransfer.fusion.losses import chamfer_distance, contain_loss


def test_contain_loss_decreases():
    base = torch.zeros(1, 1, 32, 32)
    base[:, :, 16, :] = 1.0
    good = base.clone()
    bad = torch.zeros_like(base)
    assert contain_loss(good, base) < contain_loss(bad, base)


def test_chamfer_zero():
    base = torch.zeros(1, 1, 32, 32)
    base[:, :, 16, :] = 1.0
    out = base.clone()
    assert chamfer_distance(base, out) == pytest.approx(0.0, abs=1e-6)
