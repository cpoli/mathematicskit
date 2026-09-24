"""Tests for Kahan compensated summation against exact (math.fsum) sums."""

import math

import numpy as np
import pytest

from mathematicskit.numerical_analysis.systems.summation import kahan_sum


def _naive_sum(values):
    s = 0.0
    for v in values:
        s += v
    return s


def test_kahan_recovers_small_terms_lost_by_naive_summation():
    values = [1.0] + [1e-16] * 10_000
    assert _naive_sum(values) == 1.0
    assert kahan_sum(values) == pytest.approx(math.fsum(values), rel=1e-15)


def test_kahan_error_is_independent_of_n():
    values = [0.1] * 1_000_000
    exact = math.fsum(values)
    assert abs(kahan_sum(values) - exact) <= 2 * np.finfo(float).eps * exact
    assert abs(_naive_sum(values) - exact) > 1000 * abs(kahan_sum(values) - exact)


def test_kahan_matches_fsum_on_random_data():
    x = np.random.default_rng(1).normal(size=5000) * 10.0 ** np.random.default_rng(2).integers(-8, 8, size=5000)
    assert kahan_sum(x) == pytest.approx(math.fsum(x), rel=1e-12)


def test_kahan_empty_and_multidimensional_input():
    assert kahan_sum([]) == 0.0
    assert kahan_sum(np.ones((3, 4))) == 12.0
