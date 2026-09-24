"""Tests for elliptic integrals, the AGM, and Jacobi elliptic functions against closed-form results."""

import math

import numpy as np
import pytest
from scipy import integrate

from mathematicskit.special_functions.systems.elliptic import (
    arithmetic_geometric_mean,
    complete_elliptic_integral_first_kind,
    complete_elliptic_integral_second_kind,
    jacobi_elliptic_functions,
)


def test_agm_matches_hand_iteration():
    a, b = 1.0, 3.0
    for _ in range(10):
        a, b = (a + b) / 2, math.sqrt(a * b)
    assert arithmetic_geometric_mean(1.0, 3.0) == pytest.approx(a)


def test_agm_lies_between_geometric_and_arithmetic_means():
    g = arithmetic_geometric_mean(2.0, 8.0)
    assert 4.0 < g < 5.0


@pytest.mark.parametrize("m", [0.0, 0.3, 0.7, 0.99])
def test_gauss_agm_formula_for_k(m):
    """K(m) = pi / (2 AGM(1, sqrt(1 - m)))."""
    expected = math.pi / (2 * arithmetic_geometric_mean(1.0, math.sqrt(1 - m)))
    assert complete_elliptic_integral_first_kind(m) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("m", [0.2, 0.5, 0.9])
def test_elliptic_integrals_match_quadrature(m):
    k_quad, _ = integrate.quad(lambda t: 1 / math.sqrt(1 - m * math.sin(t) ** 2), 0, math.pi / 2)
    e_quad, _ = integrate.quad(lambda t: math.sqrt(1 - m * math.sin(t) ** 2), 0, math.pi / 2)
    assert complete_elliptic_integral_first_kind(m) == pytest.approx(k_quad)
    assert complete_elliptic_integral_second_kind(m) == pytest.approx(e_quad)


def test_legendre_relation():
    """E K' + E' K - K K' = pi/2, with primes meaning complementary parameter 1 - m."""
    m = 0.37
    K, E = complete_elliptic_integral_first_kind(m), complete_elliptic_integral_second_kind(m)
    Kp, Ep = complete_elliptic_integral_first_kind(1 - m), complete_elliptic_integral_second_kind(1 - m)
    assert E * Kp + Ep * K - K * Kp == pytest.approx(math.pi / 2)


def test_jacobi_identities():
    u = np.linspace(-3, 3, 41)
    m = 0.6
    r = jacobi_elliptic_functions(u, m)
    np.testing.assert_allclose(r.sn**2 + r.cn**2, 1.0, atol=1e-13)
    np.testing.assert_allclose(r.dn**2 + m * r.sn**2, 1.0, atol=1e-13)


def test_jacobi_reduces_to_trig_and_hyperbolic():
    u = np.linspace(-2, 2, 21)
    r0 = jacobi_elliptic_functions(u, 0.0)
    np.testing.assert_allclose(r0.sn, np.sin(u), atol=1e-13)
    np.testing.assert_allclose(r0.cn, np.cos(u), atol=1e-13)
    r1 = jacobi_elliptic_functions(u, 1.0)
    np.testing.assert_allclose(r1.sn, np.tanh(u), atol=1e-12)


def test_sn_has_period_4k():
    m = 0.8
    K = complete_elliptic_integral_first_kind(m)
    u = np.linspace(0, 2, 11)
    np.testing.assert_allclose(jacobi_elliptic_functions(u + 4 * K, m).sn, jacobi_elliptic_functions(u, m).sn, atol=1e-10)
    assert jacobi_elliptic_functions(K, m).sn == pytest.approx(1.0)
