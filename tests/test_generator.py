import numpy as np
from PIL import Image

from hanzitransfer.data.generator import compose_bitmaps, generate_dataset


def test_compose_bitmaps_left_right():
    radical = Image.new('1', (64, 64), 0)
    base = Image.new('1', (64, 64), 1)
    composed = compose_bitmaps(radical, base, 'left-right')
    arr = np.array(composed)
    assert arr.shape == (64, 64)
    assert np.all(arr[:, :32] == 0)
    assert np.all(arr[:, 32:] == 1)


def test_compose_bitmaps_top_bottom():
    radical = Image.new('1', (64, 64), 0)
    base = Image.new('1', (64, 64), 1)
    composed = compose_bitmaps(radical, base, 'top-bottom')
    arr = np.array(composed)
    assert arr.shape == (64, 64)
    assert np.all(arr[:32, :] == 0)
    assert np.all(arr[32:, :] == 1)


def test_generate_dataset_shapes():
    chars = ['口']
    radicals = ['氵']
    base, target = generate_dataset(radicals=radicals, chars=chars, limit=1)
    assert base.shape == (1, 64, 64)
    assert target.shape == (1, 64, 64)
    assert not np.array_equal(base[0], target[0])
