from utils.audio import zero_samples
import numpy as np
from types import SimpleNamespace


def test_zero_samples():
    ones = np.ones(100)
    args = SimpleNamespace()
    args.zero_rate = 0.9
    zeroed = zero_samples(args, ones)
    assert zeroed.sum() == 10
    assert ones.sum() == 100

    args = SimpleNamespace()
    args.zero_rate = 0.1
    zeroed = zero_samples(args, ones)
    assert zeroed.sum() == 90
    assert all(np.convolve(zeroed, np.ones(10), mode="valid") > 5)
