"""Tests for descriptive statistics against closed-form/known results."""

import numpy as np
import pytest

from mathkit.statistics.systems.descriptive import descriptive_stats, order_statistic


def test_matches_hand_computed_five_number_summary():
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    result = descriptive_stats(data)
    assert result.n == 8
    assert result.mean == pytest.approx(5.0)
    assert result.std == pytest.approx(2.13809, abs=1e-4)
    assert result.minimum == 2.0
    assert result.maximum == 9.0
    assert result.median == pytest.approx(4.5)


def test_symmetric_data_has_zero_skewness():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = descriptive_stats(data)
    assert result.skewness == pytest.approx(0.0, abs=1e-8)


def test_variance_matches_numpy_ddof1():
    rng = np.random.default_rng(0)
    data = rng.normal(size=500)
    result = descriptive_stats(data)
    assert result.variance == pytest.approx(np.var(data, ddof=1))


def test_order_statistic_matches_sorted_array():
    data = [5.0, 1.0, 3.0, 2.0, 4.0]
    for k in range(1, 6):
        assert order_statistic(data, k) == float(k)


def test_order_statistic_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        order_statistic([1.0, 2.0, 3.0], k=0)
    with pytest.raises(ValueError):
        order_statistic([1.0, 2.0, 3.0], k=4)
