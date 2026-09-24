"""Tests for the error function and Fresnel integrals against closed-form results."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from mathematicskit.special_functions.systems.error_functions import complementary_error_function, error_function, fresnel_integrals


def test_erf_limits_and_oddness():
    assert error_function(0.0) == 0.0
    assert error_function(10.0) == pytest.approx(1.0)
    x = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(error_function(-x), -error_function(x))


def test_erf_matches_its_integral():
    x = 1.3
    integral, _ = integrate.quad(lambda t: math.exp(-(t**2)), 0, x)
    assert error_function(x) == pytest.approx(2 / math.sqrt(math.pi) * integral)


@pytest.mark.parametrize(("k", "prob"), [(1, 0.682689492), (2, 0.954499736), (3, 0.997300204)])
def test_68_95_997_rule(k, prob):
    assert error_function(k / math.sqrt(2)) == pytest.approx(prob, abs=1e-9)


def test_normal_cdf_relation():
    x = np.linspace(-4, 4, 17)
    np.testing.assert_allclose(0.5 * complementary_error_function(-x / np.sqrt(2)), stats.norm.cdf(x), rtol=1e-12)


def test_erfc_keeps_precision_in_tail():
    assert 1.0 - error_function(10.0) == 0.0
    # erfc(x) ~ exp(-x^2) / (x sqrt(pi)) (1 - 1/(2x^2))
    x = 10.0
    asymptotic = math.exp(-(x**2)) / (x * math.sqrt(math.pi)) * (1 - 1 / (2 * x**2))
    assert complementary_error_function(x) == pytest.approx(asymptotic, rel=1e-3)


def test_fresnel_limits_are_one_half():
    r = fresnel_integrals(np.array([1e5, -1e5]))
    np.testing.assert_allclose(r.c, [0.5, -0.5], atol=1e-5)
    np.testing.assert_allclose(r.s, [0.5, -0.5], atol=1e-5)


def test_fresnel_derivatives():
    """dC/dt = cos(pi t^2 / 2), dS/dt = sin(pi t^2 / 2)."""
    t, h = 1.7, 1e-5
    plus, minus = fresnel_integrals(t + h), fresnel_integrals(t - h)
    assert (plus.c - minus.c) / (2 * h) == pytest.approx(math.cos(math.pi * t**2 / 2), abs=1e-8)
    assert (plus.s - minus.s) / (2 * h) == pytest.approx(math.sin(math.pi * t**2 / 2), abs=1e-8)


def test_cornu_spiral_has_unit_speed_and_curvature_proportional_to_arclength():
    """The Euler spiral (C(t), S(t)) is parametrized by arclength with curvature pi t."""
    t = np.linspace(0.2, 2.0, 2001)
    r = fresnel_integrals(t)
    dc, ds = np.gradient(r.c, t), np.gradient(r.s, t)
    np.testing.assert_allclose(np.hypot(dc, ds)[5:-5], 1.0, atol=1e-5)
    d2c, d2s = np.gradient(dc, t), np.gradient(ds, t)
    curvature = dc * d2s - ds * d2c
    np.testing.assert_allclose(curvature[10:-10], np.pi * t[10:-10], rtol=1e-3)
