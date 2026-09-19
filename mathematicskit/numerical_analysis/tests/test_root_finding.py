"""Tests for root finders against closed-form/analytically-known roots and
their theoretical convergence orders.
"""

import numpy as np
import pytest
from scipy import optimize

from mathematicskit.numerical_analysis.systems.root_finding import Bisection, FixedPointIteration, NewtonRaphson, Secant
from mathematicskit.numerical_analysis.utils.error_analysis import estimate_convergence_order


def test_bisection_finds_sqrt2():
    result = Bisection(lambda x: x**2 - 2.0, 0.0, 2.0, tol=1e-12).solve()
    assert result.converged
    assert result.root == pytest.approx(np.sqrt(2.0), abs=1e-9)


def test_bisection_rejects_bracket_without_sign_change():
    with pytest.raises(ValueError):
        Bisection(lambda x: x**2 + 1.0, 0.0, 2.0)


def test_bisection_handles_root_at_endpoint():
    result = Bisection(lambda x: x - 1.0, 1.0, 2.0).solve()
    assert result.root == pytest.approx(1.0)
    assert result.converged


def test_newton_raphson_converges_quadratically_to_sqrt2():
    result = NewtonRaphson(lambda x: x**2 - 2.0, lambda x: 2.0 * x, x0=1.0, tol=1e-14).solve()
    assert result.root == pytest.approx(np.sqrt(2.0), abs=1e-12)
    order = estimate_convergence_order(result.history, np.sqrt(2.0))
    assert order == pytest.approx(2.0, abs=0.3)


def test_newton_raphson_rejects_zero_derivative():
    with pytest.raises(ZeroDivisionError):
        NewtonRaphson(lambda x: x**2, lambda x: 2.0 * x, x0=0.0).solve()


def test_secant_finds_sqrt2_without_a_derivative():
    result = Secant(lambda x: x**2 - 2.0, x0=1.0, x1=2.0, tol=1e-14).solve()
    assert result.root == pytest.approx(np.sqrt(2.0), abs=1e-10)


def test_secant_converges_superlinearly():
    result = Secant(lambda x: x**2 - 2.0, x0=1.0, x1=2.0, tol=1e-15, max_iter=200).solve()
    order = estimate_convergence_order(result.history, np.sqrt(2.0))
    # Theoretical order is the golden ratio ~1.618; allow a generous band.
    assert 1.2 < order < 2.2


def test_fixed_point_iteration_solves_x_equals_cos_x():
    result = FixedPointIteration(np.cos, x0=0.5, tol=1e-12).solve()
    assert result.converged
    assert result.root == pytest.approx(np.cos(result.root), abs=1e-9)


def test_all_methods_agree_on_the_same_root():
    """Bisection, Newton, secant, and fixed-point iteration solving
    (differently posed) equivalent problems should land on the same
    root, sqrt(2)."""
    root_bisect = Bisection(lambda x: x**2 - 2.0, 1.0, 2.0).solve().root
    root_newton = NewtonRaphson(lambda x: x**2 - 2.0, lambda x: 2.0 * x, x0=1.5).solve().root
    root_secant = Secant(lambda x: x**2 - 2.0, x0=1.0, x1=2.0).solve().root
    root_fixed = FixedPointIteration(lambda x: 0.5 * (x + 2.0 / x), x0=1.5).solve().root  # Babylonian method
    for root in (root_bisect, root_newton, root_secant, root_fixed):
        assert root == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_bisection_matches_scipy_brentq():
    """Cross-check the hand-rolled bisection root against scipy.optimize.brentq
    -- the from-scratch iteration is kept for its exposed per-iterate history
    (see estimate_convergence_order), not because scipy lacks a bracketed
    root finder."""
    f = lambda x: x**3 - x - 2.0
    root_mathematicskit = Bisection(f, 1.0, 2.0, tol=1e-12).solve().root
    root_scipy = optimize.brentq(f, 1.0, 2.0, xtol=1e-12)
    assert root_mathematicskit == pytest.approx(root_scipy, abs=1e-9)


def test_newton_raphson_matches_scipy_newton():
    f = lambda x: x**2 - 2.0
    fprime = lambda x: 2.0 * x
    root_mathematicskit = NewtonRaphson(f, fprime, x0=1.0, tol=1e-14).solve().root
    root_scipy = optimize.newton(f, x0=1.0, fprime=fprime, tol=1e-14)
    assert root_mathematicskit == pytest.approx(root_scipy, abs=1e-12)
