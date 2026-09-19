"""Tests for Lagrange and Newton divided-difference interpolation: both
must reproduce any polynomial up to the interpolation degree exactly
(closed-form correctness), and must agree with each other everywhere.
"""

import numpy as np
import pytest
from scipy import interpolate

from mathematicskit.numerical_analysis.systems.interpolation import LagrangeInterpolant, NewtonDividedDifference


def test_lagrange_is_exact_for_polynomial_up_to_degree():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x**3 - x**2 + 5.0
    p = LagrangeInterpolant(x, y)
    x_test = np.linspace(-1.0, 5.0, 25)
    expected = 2.0 * x_test**3 - x_test**2 + 5.0
    np.testing.assert_allclose(p.evaluate(x_test), expected, atol=1e-8)


def test_newton_divided_difference_is_exact_for_polynomial_up_to_degree():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x**3 - x**2 + 5.0
    p = NewtonDividedDifference(x, y)
    x_test = np.linspace(-1.0, 5.0, 25)
    expected = 2.0 * x_test**3 - x_test**2 + 5.0
    np.testing.assert_allclose(p.evaluate(x_test), expected, atol=1e-8)


def test_lagrange_and_newton_agree():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-3.0, 3.0, 8))
    y = np.sin(x) + rng.normal(0.0, 0.01, x.shape)
    p_lagrange = LagrangeInterpolant(x, y)
    p_newton = NewtonDividedDifference(x, y)
    x_test = np.linspace(x.min(), x.max(), 40)
    np.testing.assert_allclose(p_lagrange.evaluate(x_test), p_newton.evaluate(x_test), atol=1e-8)


def test_interpolant_reproduces_nodes_exactly():
    x = np.array([0.0, 1.0, 2.5, 4.0])
    y = np.array([1.0, -2.0, 3.0, 0.5])
    for cls in (LagrangeInterpolant, NewtonDividedDifference):
        p = cls(x, y)
        np.testing.assert_allclose(p.evaluate(x), y, atol=1e-10)


def test_rejects_duplicate_nodes():
    with pytest.raises(ValueError):
        LagrangeInterpolant([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        NewtonDividedDifference([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])


def test_scalar_input_returns_scalar_output():
    p = LagrangeInterpolant([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    value = p.evaluate(1.5)
    assert isinstance(value, float)


def test_lagrange_and_newton_agree_with_scipy_barycentric_interpolator():
    """Cross-check against scipy.interpolate.BarycentricInterpolator --
    the hand-rolled Lagrange/Newton forms are kept because their explicit
    basis/table construction is the pedagogical point, not because scipy
    lacks a polynomial interpolator."""
    rng = np.random.default_rng(6)
    x = np.sort(rng.uniform(-3.0, 3.0, 7))
    y = np.sin(x)
    x_test = np.linspace(x.min(), x.max(), 30)

    p_lagrange = LagrangeInterpolant(x, y)
    p_newton = NewtonDividedDifference(x, y)
    p_scipy = interpolate.BarycentricInterpolator(x, y)

    np.testing.assert_allclose(p_lagrange.evaluate(x_test), p_scipy(x_test), atol=1e-8)
    np.testing.assert_allclose(p_newton.evaluate(x_test), p_scipy(x_test), atol=1e-8)
