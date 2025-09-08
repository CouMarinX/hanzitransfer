import numpy as np

from hanzitransfer.inference.renderer import render_strokes


def test_render_strokes_line_pixel_count():
    strokes = [
        [
            {"op": "M", "points": [(0, 0)]},
            {"op": "L", "points": [(63, 0)]},
        ]
    ]
    img = render_strokes(strokes, width=1)
    arr = np.array(img)
    assert arr.shape == (64, 64)
    assert np.count_nonzero(arr == 0) == 64
