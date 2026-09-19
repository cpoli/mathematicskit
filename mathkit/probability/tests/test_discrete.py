"""Tests for discrete distributions against closed-form mean/variance
and MGF derivative identities."""

import numpy as np
import pytest

from mathkit.probability.systems.discrete import Binomial, Geometric, Poisson


def _mgf_derivative(dist, order, h=1e-4):
    """Numerically differentiate the MGF at t=0 to cross-check E[X] (order=1)
    and E[X^2] (order=2) against the closed-form mean/variance."""
    if order == 1:
        return (dist.mgf(h) - dist.mgf(-h)) / (2 * h)
    if order == 2:
        return (dist.mgf(h) - 2 * dist.mgf(0.0) + dist.mgf(-h)) / h**2
    raise ValueError


def test_binomial_mean_and_variance_match_closed_form():
    b = Binomial(n=20, p=0.4)
    assert b.mean == pytest.approx(20 * 0.4)
    assert b.variance == pytest.approx(20 * 0.4 * 0.6)


def test_binomial_pmf_sums_to_one():
    b = Binomial(n=15, p=0.6)
    ks = np.arange(0, 16)
    assert np.sum(b.pmf(ks)) == pytest.approx(1.0, abs=1e-8)


def test_binomial_mgf_derivative_matches_mean():
    b = Binomial(n=10, p=0.3)
    assert _mgf_derivative(b, 1) == pytest.approx(b.mean, abs=1e-4)
    second_moment = _mgf_derivative(b, 2)
    assert second_moment - b.mean**2 == pytest.approx(b.variance, abs=1e-3)


def test_poisson_mean_equals_variance():
    p = Poisson(mu=7.0)
    assert p.mean == pytest.approx(7.0)
    assert p.variance == pytest.approx(7.0)


def test_poisson_mgf_derivative_matches_mean():
    p = Poisson(mu=3.0)
    assert _mgf_derivative(p, 1) == pytest.approx(p.mean, abs=1e-4)


def test_geometric_mean_matches_closed_form():
    g = Geometric(p=0.2)
    assert g.mean == pytest.approx(1.0 / 0.2)
    assert g.variance == pytest.approx((1.0 - 0.2) / 0.2**2)


def test_geometric_mgf_at_zero_is_one():
    g = Geometric(p=0.5)
    assert g.mgf(0.0) == pytest.approx(1.0)


def test_sample_shape_and_reproducibility():
    b = Binomial(n=10, p=0.5)
    a = b.sample(size=1000, seed=0)
    c = b.sample(size=1000, seed=0)
    np.testing.assert_array_equal(a, c)
    assert a.shape == (1000,)
