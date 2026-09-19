"""Tests for confidence intervals against closed-form coverage properties."""

import numpy as np
import pytest

from mathematicskit.statistics.systems.confidence_intervals import mean_confidence_interval, proportion_confidence_interval, variance_confidence_interval


def test_mean_ci_with_known_sigma_covers_true_mean_most_of_the_time():
    rng = np.random.default_rng(0)
    true_mean = 10.0
    covered = 0
    trials = 200
    for _ in range(trials):
        data = rng.normal(loc=true_mean, scale=2.0, size=50)
        result = mean_confidence_interval(data, sigma=2.0, confidence_level=0.95)
        if result.lower < true_mean < result.upper:
            covered += 1
    assert covered / trials > 0.85  # should be close to 0.95; generous bound to avoid flakiness


def test_mean_ci_t_based_is_wider_than_z_based():
    """With the same standard deviation plugged into both formulas (the
    sample std, so only the critical-value choice differs), the
    heavier-tailed t distribution at a small df must give a wider
    interval than the normal approximation."""
    rng = np.random.default_rng(1)
    data = rng.normal(loc=0.0, scale=1.0, size=10)
    sample_std = np.std(data, ddof=1)
    result_z = mean_confidence_interval(data, sigma=sample_std)
    result_t = mean_confidence_interval(data, sigma=None)
    assert result_t.width() > result_z.width()


def test_proportion_ci_matches_hand_computed_value():
    result = proportion_confidence_interval(successes=520, n=1000)
    assert result.estimate == pytest.approx(0.52)
    p_hat = 0.52
    z = 1.959963984540054  # 97.5th percentile of N(0,1)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / 1000)
    assert result.lower == pytest.approx(p_hat - margin, abs=1e-6)
    assert result.upper == pytest.approx(p_hat + margin, abs=1e-6)


def test_variance_ci_covers_true_variance_most_of_the_time():
    rng = np.random.default_rng(2)
    true_var = 9.0
    covered = 0
    trials = 200
    for _ in range(trials):
        data = rng.normal(loc=0.0, scale=3.0, size=50)
        result = variance_confidence_interval(data, confidence_level=0.95)
        if result.lower < true_var < result.upper:
            covered += 1
    assert covered / trials > 0.85


def test_narrower_confidence_level_gives_narrower_interval():
    rng = np.random.default_rng(3)
    data = rng.normal(size=100)
    wide = mean_confidence_interval(data, sigma=1.0, confidence_level=0.99)
    narrow = mean_confidence_interval(data, sigma=1.0, confidence_level=0.80)
    assert narrow.width() < wide.width()
