"""Tests for continuous distributions against closed-form mean/variance
and MGF derivative identities."""

import numpy as np
import pytest
from scipy import integrate

from mathkit.probability.systems.continuous import Exponential, Gamma, Normal, Uniform


def _mgf_derivative(dist, order, h=1e-4):
    if order == 1:
        return (dist.mgf(h) - dist.mgf(-h)) / (2 * h)
    if order == 2:
        return (dist.mgf(h) - 2 * dist.mgf(0.0) + dist.mgf(-h)) / h**2
    raise ValueError


def test_uniform_mean_and_variance_match_closed_form():
    u = Uniform(a=1.0, b=5.0)
    assert u.mean == pytest.approx(3.0)
    assert u.variance == pytest.approx((5.0 - 1.0) ** 2 / 12.0)


def test_uniform_pdf_integrates_to_one():
    u = Uniform(a=-2.0, b=3.0)
    value, _ = integrate.quad(u.pdf, -2.0, 3.0)
    assert value == pytest.approx(1.0, abs=1e-8)


def test_uniform_mgf_at_zero_is_one():
    u = Uniform(a=0.0, b=1.0)
    assert u.mgf(0.0) == pytest.approx(1.0)


def test_uniform_mgf_derivative_matches_mean():
    u = Uniform(a=0.0, b=4.0)
    assert _mgf_derivative(u, 1) == pytest.approx(u.mean, abs=1e-3)


def test_exponential_mean_and_variance_match_closed_form():
    e = Exponential(rate=2.5)
    assert e.mean == pytest.approx(1.0 / 2.5)
    assert e.variance == pytest.approx(1.0 / 2.5**2)


def test_exponential_mgf_derivative_matches_mean():
    e = Exponential(rate=3.0)
    assert _mgf_derivative(e, 1) == pytest.approx(e.mean, abs=1e-4)


def test_normal_mgf_derivative_matches_mean_and_variance():
    n = Normal(mu=2.0, sigma=1.5)
    assert _mgf_derivative(n, 1) == pytest.approx(n.mean, abs=1e-4)
    second_moment = _mgf_derivative(n, 2)
    assert second_moment - n.mean**2 == pytest.approx(n.variance, abs=1e-3)


def test_normal_rejects_non_positive_sigma():
    with pytest.raises(ValueError):
        Normal(mu=0.0, sigma=0.0)


def test_gamma_mean_and_variance_match_closed_form():
    g = Gamma(shape=3.0, rate=2.0)
    assert g.mean == pytest.approx(3.0 / 2.0)
    assert g.variance == pytest.approx(3.0 / 2.0**2)


def test_gamma_with_shape_one_matches_exponential():
    g = Gamma(shape=1.0, rate=2.0)
    e = Exponential(rate=2.0)
    xs = np.linspace(0.01, 5.0, 20)
    np.testing.assert_allclose(g.pdf(xs), e.pdf(xs), atol=1e-8)
