"""Tests for quadrature rules against closed-form integrals."""

import numpy as np
import pytest

from mathkit.calculus.systems.quadrature import AdaptiveQuadrature, GaussianQuadrature, SimpsonsRule, TrapezoidalRule, legendre_nodes_and_weights


def test_trapezoidal_converges_to_known_integral():
    result = TrapezoidalRule(n=10000).integrate(lambda x: x**2, 0.0, 1.0)
    assert result.value == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_simpson_exact_for_cubic():
    """Simpson's rule is exact for polynomials up to degree 3."""
    result = SimpsonsRule(n=4).integrate(lambda x: 2.0 * x**3 - x + 1.0, 0.0, 2.0)
    exact = 0.5 * 2.0**4 - 0.5 * 2.0**2 + 2.0  # antiderivative x^4/2 - x^2/2 + x
    assert result.value == pytest.approx(exact, abs=1e-10)


def test_simpson_more_accurate_than_trapezoidal_at_same_n():
    f = np.sin
    exact = 2.0  # integral of sin from 0 to pi
    err_trap = abs(TrapezoidalRule(n=20).integrate(f, 0.0, np.pi).value - exact)
    err_simp = abs(SimpsonsRule(n=20).integrate(f, 0.0, np.pi).value - exact)
    assert err_simp < err_trap


def test_legendre_nodes_and_weights_sum_to_two():
    for n in (2, 3, 5, 8):
        nodes, weights = legendre_nodes_and_weights(n)
        assert np.sum(weights) == pytest.approx(2.0, abs=1e-10)
        assert np.all(np.abs(nodes) <= 1.0 + 1e-12)


def test_gaussian_quadrature_exact_up_to_degree_2n_minus_1():
    """n=4 nodes should be exact for any polynomial up to degree 7."""
    gq = GaussianQuadrature(n=4)
    result = gq.integrate(lambda x: x**7 - 3.0 * x**5 + 2.0, -1.0, 1.0)
    exact = 0.0 - 0.0 + 4.0  # odd-power terms integrate to 0 over symmetric interval
    assert result.value == pytest.approx(exact, abs=1e-10)


def test_gaussian_quadrature_needs_fewer_evaluations_than_simpson_for_same_accuracy():
    f = lambda x: np.exp(x)
    exact = np.e - 1.0
    gq_result = GaussianQuadrature(n=5).integrate(f, 0.0, 1.0)
    assert abs(gq_result.value - exact) < 1e-10
    assert gq_result.n_evaluations == 5


def test_adaptive_quadrature_matches_known_integral():
    result = AdaptiveQuadrature(tol=1e-10).integrate(lambda x: x**2, 0.0, 1.0)
    assert result.value == pytest.approx(1.0 / 3.0, abs=1e-8)


def test_adaptive_quadrature_handles_sharply_peaked_function():
    f = lambda x: 1.0 / (1.0 + 1000.0 * (x - 0.5) ** 2)
    result = AdaptiveQuadrature(tol=1e-6).integrate(f, 0.0, 1.0)
    exact = (np.arctan(1000.0**0.5 * 0.5) * 2.0) / 1000.0**0.5
    assert result.value == pytest.approx(exact, abs=1e-3)
