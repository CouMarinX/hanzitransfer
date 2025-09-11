import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from hanzitransfer.fusion.infer_fusion import main as infer_main
from hanzitransfer.fusion.models.cond_unet import CondUNet


def test_infer_cli(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    model = CondUNet(in_channels=4)
    torch.save(model.state_dict(), ckpt)
    outdir = tmp_path / "outs"
    infer_main([
        "--base",
        "木",
        "--layout",
        "⿰",
        "--num",
        "2",
        "--ckpt",
        str(ckpt),
        "--outdir",
        str(outdir),
    ])
    assert (outdir / "sample_0.png").exists()
    assert (outdir / "grid.png").exists()
