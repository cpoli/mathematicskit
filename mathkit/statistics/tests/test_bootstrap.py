"""Tests for bootstrap confidence intervals against closed-form coverage
and against the parametric interval they approximate."""

import numpy as np
import pytest

from mathkit.statistics.systems.bootstrap import bootstrap_confidence_interval
from mathkit.statistics.systems.confidence_intervals import mean_confidence_interval


def test_bootstrap_mean_ci_covers_true_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=300)
    result = bootstrap_confidence_interval(data, statistic=np.mean, n_resamples=2000, seed=0)
    assert result.lower < 5.0 < result.upper
    assert result.estimate == pytest.approx(np.mean(data))


def test_bootstrap_mean_ci_roughly_matches_parametric_t_interval():
    rng = np.random.default_rng(1)
    data = rng.normal(loc=0.0, scale=1.0, size=500)
    bootstrap_result = bootstrap_confidence_interval(data, statistic=np.mean, n_resamples=3000, seed=1)
    parametric_result = mean_confidence_interval(data, sigma=None)
    assert bootstrap_result.lower == pytest.approx(parametric_result.lower, abs=0.05)
    assert bootstrap_result.upper == pytest.approx(parametric_result.upper, abs=0.05)


def test_bootstrap_std_error_decreases_with_more_data():
    rng = np.random.default_rng(2)
    small_data = rng.normal(size=30)
    large_data = rng.normal(size=3000)
    small_result = bootstrap_confidence_interval(small_data, n_resamples=2000, seed=2)
    large_result = bootstrap_confidence_interval(large_data, n_resamples=2000, seed=2)
    assert large_result.std_error < small_result.std_error


def test_bootstrap_works_with_median_statistic():
    rng = np.random.default_rng(3)
    data = rng.normal(loc=3.0, size=300)
    result = bootstrap_confidence_interval(data, statistic=np.median, n_resamples=2000, seed=3)
    assert result.lower < 3.0 < result.upper
