import numpy as np

from hanzitransfer.fusion.vector_geom import (
    Path,
    paths_curvature_stats,
    paths_self_intersections,
)


def test_self_intersection_count():
    cross = [
        Path(np.array([[0, 0], [1, 1]], dtype=float)),
        Path(np.array([[0, 1], [1, 0]], dtype=float)),
    ]
    assert paths_self_intersections(cross) == 1

    square = [Path(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))]
    assert paths_self_intersections(square) == 0


def test_curvature_stats():
    path = Path(np.array([[0, 0], [1, 0], [1, 1]], dtype=float))
    stats = paths_curvature_stats([path])
    assert stats["length"] > 0
    assert stats["max_curvature"] >= stats["mean_curvature"] > 0

