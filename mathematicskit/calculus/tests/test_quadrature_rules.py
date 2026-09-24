"""Tests for Riemann sums, Romberg, Clenshaw-Curtis, tanh-sinh, and the
Euler-Maclaurin-corrected trapezoidal rule against closed-form integrals."""

import numpy as np
import pytest
from scipy import integrate

from mathematicskit.calculus.systems.quadrature import (
    ClenshawCurtisQuadrature,
    RiemannSum,
    RombergQuadrature,
    SimpsonsRule,
    TanhSinhQuadrature,
    clenshaw_curtis_nodes_and_weights,
    euler_maclaurin_trapezoid,
)

EXACT_EXP = np.e - 1


@pytest.mark.parametrize("rule,order", [("left", 1), ("right", 1), ("midpoint", 2)])
def test_riemann_sum_convergence_order(rule, order):
    errors = [abs(RiemannSum(n, rule).integrate(np.exp, 0.0, 1.0).value - EXACT_EXP) for n in (100, 200)]
    assert errors[0] / errors[1] == pytest.approx(2**order, rel=0.02)


def test_riemann_left_and_right_bracket_an_increasing_integrand():
    left = RiemannSum(50, "left").integrate(np.exp, 0.0, 1.0).value
    right = RiemannSum(50, "right").integrate(np.exp, 0.0, 1.0).value
    assert left < EXACT_EXP < right


def test_riemann_rejects_unknown_rule():
    with pytest.raises(ValueError):
        RiemannSum(10, "trapezoid")


def test_simpson_is_exact_for_cubics():
    result = SimpsonsRule(n=2).integrate(lambda x: 4 * x**3 - x + 2, 0.0, 2.0)
    assert result.value == pytest.approx(16 - 2 + 4)


def test_romberg_matches_exact_integral_and_first_column_is_trapezoid():
    result = RombergQuadrature(levels=6).integrate(np.exp, 0.0, 1.0)
    assert result.value == pytest.approx(EXACT_EXP, abs=1e-14)
    assert result.extra["table"][1][0] == pytest.approx(0.25 * (1 + 2 * np.exp(0.5) + np.e))
    assert result.extra["table"][1][1] == pytest.approx(integrate.simpson(np.exp([0, 0.5, 1]), x=[0, 0.5, 1]))


def test_romberg_is_exact_for_polynomials_of_matching_degree():
    result = RombergQuadrature(levels=3).integrate(lambda x: x**7, 0.0, 1.0)
    assert result.value == pytest.approx(1 / 8, abs=1e-14)


@pytest.mark.parametrize("n", [2, 3, 8, 9, 16])
def test_clenshaw_curtis_weights_are_positive_and_exact_on_polynomials(n):
    x, w = clenshaw_curtis_nodes_and_weights(n)
    assert np.all(w > 0)
    for degree in range(n + 1):
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert np.dot(w, x**degree) == pytest.approx(exact, abs=1e-13)


def test_clenshaw_curtis_matches_scipy_quad_on_smooth_function():
    f = lambda x: np.cos(3 * x) * np.exp(-x)  # noqa: E731
    reference, _ = integrate.quad(f, -1.0, 2.0)
    assert ClenshawCurtisQuadrature(32).integrate(f, -1.0, 2.0).value == pytest.approx(reference, abs=1e-13)


@pytest.mark.parametrize(
    "f,a,b,exact,tol",
    [
        (lambda x: 1 / np.sqrt(x), 0.0, 1.0, 2.0, 1e-9),
        (np.log, 0.0, 1.0, -1.0, 1e-9),
        (np.exp, 0.0, 1.0, np.e - 1, 1e-12),
        # 1 - x**2 loses digits once x is rounded near +-1, capping the accuracy.
        (lambda x: 1 / np.sqrt(1 - x**2), -1.0, 1.0, np.pi, 1e-7),
    ],
)
def test_tanh_sinh_handles_endpoint_singularities(f, a, b, exact, tol):
    assert TanhSinhQuadrature(h=0.05).integrate(f, a, b).value == pytest.approx(exact, abs=tol)


def test_euler_maclaurin_corrections_raise_the_order():
    exact = EXACT_EXP
    errors = []
    for m in range(4):
        result = euler_maclaurin_trapezoid(np.exp, 0.0, 1.0, 8, [np.exp] * m)
        errors.append(abs(result.value - exact))
    assert all(a > 100 * b for a, b in zip(errors, errors[1:3]))
