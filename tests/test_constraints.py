import numpy as np

from hanzitransfer.fusion.constraints import Proj_C
from hanzitransfer.fusion.vector_geom import Path, paths_self_intersections, rasterize_paths


def _containment(base, pred):
    base_r = rasterize_paths(base, 10, 10) > 0
    pred_r = rasterize_paths(pred, 10, 10) > 0
    return (base_r & pred_r).sum() / max(1, base_r.sum())


def test_proj_c_reduces_intersections_and_increases_containment():
    base = [Path(np.array([[1, 1], [8, 1], [8, 8], [1, 8]], dtype=float))]
    cross = [
        Path(np.array([[0, 0], [9, 9]], dtype=float)),
        Path(np.array([[0, 9], [9, 0]], dtype=float)),
    ]
    inter_before = paths_self_intersections(cross)
    contain_before = _containment(base, cross)

    out = Proj_C(cross, base, "⿰", 0.5)
    inter_after = paths_self_intersections(out)
    contain_after = _containment(base, out)

    assert inter_after < inter_before
    assert contain_after > contain_before

