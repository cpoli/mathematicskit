"""Tests for Lagrange and Newton divided-difference interpolation: both
must reproduce any polynomial up to the interpolation degree exactly
(closed-form correctness), and must agree with each other everywhere.
"""

import numpy as np
import pytest
from scipy import interpolate

from mathematicskit.numerical_analysis.systems.interpolation import HermiteInterpolant, LagrangeInterpolant, NewtonDividedDifference


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


def test_hermite_reproduces_polynomials_up_to_degree_2n_plus_1():
    """n + 1 = 3 nodes with slopes fix a quintic exactly."""
    p = np.poly1d([2.0, -1.0, 0.5, 3.0, -2.0, 1.0])
    x = np.array([-1.0, 0.3, 1.5])
    H = HermiteInterpolant(x, p(x), dydx=p.deriv()(x))
    x_test = np.linspace(-1.0, 1.5, 40)
    np.testing.assert_allclose(H(x_test), p(x_test), atol=1e-10)
    np.testing.assert_allclose(H.derivative(x_test), p.deriv()(x_test), atol=1e-9)


def test_hermite_matches_values_and_slopes_at_unsorted_nodes():
    x = np.array([2.0, 0.0, 1.0])
    y, dy = np.sin(x), np.cos(x)
    H = HermiteInterpolant(x, y, dydx=dy)
    np.testing.assert_allclose(H(x), y, atol=1e-12)
    np.testing.assert_allclose(H.derivative(x), dy, atol=1e-10)


def test_hermite_error_matches_closed_form_remainder():
    """For f = x^4 with nodes 0, 1 the cubic Hermite remainder is
    f^(4)/4! * x^2 (x - 1)^2 = x^2 (x - 1)^2 exactly."""
    x = np.array([0.0, 1.0])
    H = HermiteInterpolant(x, x**4, dydx=4.0 * x**3)
    t = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(t**4 - H(t), t**2 * (t - 1.0) ** 2, atol=1e-12)


def test_hermite_rejects_repeated_nodes():
    with pytest.raises(ValueError):
        HermiteInterpolant([0.0, 0.0], [1.0, 1.0], dydx=[0.0, 0.0])
