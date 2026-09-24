"""Tests for the James-Stein estimator."""

import numpy as np
import pytest

from mathematicskit.statistics.systems.shrinkage import james_stein_estimator


def test_james_stein_closed_form_factor():
    x = np.array([3.0, 4.0, 0.0, 0.0, 0.0])
    assert james_stein_estimator(x, sigma=2.0) == pytest.approx((1 - 3 * 4 / 25) * x)


def test_positive_part_clips_at_target():
    x = np.array([0.1, -0.2, 0.1])
    assert james_stein_estimator(x) == pytest.approx(np.zeros(3))
    target = np.array([1.0, 1.0, 1.0])
    assert james_stein_estimator(target + x, target=target) == pytest.approx(target)
    assert not np.allclose(james_stein_estimator(x, positive_part=False), 0.0)


def test_james_stein_beats_raw_observation_in_mean_squared_error():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=10)
    raw, shrunk = 0.0, 0.0
    for _ in range(2000):
        x = theta + rng.normal(size=10)
        raw += np.sum((x - theta) ** 2)
        shrunk += np.sum((james_stein_estimator(x) - theta) ** 2)
    assert raw / 2000 == pytest.approx(10.0, rel=0.05)
    assert shrunk < raw


def test_james_stein_needs_three_dimensions():
    with pytest.raises(ValueError):
        james_stein_estimator(np.array([1.0, 2.0]))
