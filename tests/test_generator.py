import numpy as np

from hanzitransfer.data.generator import generate_dataset


def test_generate_dataset_shapes():
    chars = ['口']
    radicals = ['氵']
    base, target = generate_dataset(radicals=radicals, chars=chars, limit=1)
    assert base.shape == (1, 64, 64)
    assert target.shape == (1, 64, 64)
    assert not np.array_equal(base[0], target[0])
