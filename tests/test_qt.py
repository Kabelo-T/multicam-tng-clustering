import numpy as np
import pytest
from qt import qt


def test_hand_example():
    # x's ranks are [2, 0, 1], so each x takes the y value with the same rank.
    np.testing.assert_allclose(qt([3, 1, 2], [10, 20, 30]), [30, 10, 20])


def test_identity():
    # matching x onto itself should return x
    x = np.random.default_rng(0).normal(size=100)
    np.testing.assert_allclose(qt(x, x), x)


def test_keeps_x_order_and_takes_y_distribution():
    # check that x keeps its ordering, y's values are properly maintained
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=500), rng.lognormal(size=500)
    out = qt(x, y)
    np.testing.assert_array_equal(np.argsort(out), np.argsort(x))
    np.testing.assert_allclose(np.sort(out), np.sort(y))


def test_different_lengths_match_quantiles():
    # x's min/median/max map to y's min/median/max.
    np.testing.assert_allclose(qt([1, 2, 3], [0, 1, 2, 3, 4]), [0, 2, 4])


def test_single_x_fails():
    # Known limitation: rank rescaling divides by len(x) - 1.
    with pytest.raises(ZeroDivisionError):
        qt([1.0], [1, 2, 3])
