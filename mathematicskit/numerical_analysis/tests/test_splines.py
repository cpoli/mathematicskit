"""Tests for cubic spline interpolation against closed-form properties:
node reproduction, natural boundary conditions, and exact reproduction of
a line/cubic by a clamped spline with matching end-slopes.
"""

import numpy as np
import pytest

from mathematicskit.numerical_analysis.systems.splines import CubicSpline


def test_natural_spline_reproduces_nodes():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    spline = CubicSpline(x, y, boundary="natural")
    np.testing.assert_allclose(spline.evaluate(x), y, atol=1e-10)


def test_natural_spline_has_zero_second_derivative_at_ends():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 5.0])
    spline = CubicSpline(x, y, boundary="natural")
    second_derivs = spline.second_derivative_at_nodes()
    assert second_derivs[0] == pytest.approx(0.0, abs=1e-10)
    assert second_derivs[-1] == pytest.approx(0.0, abs=1e-10)


def test_clamped_spline_reproduces_a_line_exactly():
    """A cubic spline clamped to a line's own end-slopes must reproduce
    the line everywhere -- the unique interpolating cubic spline for
    linear data with matching end-slopes has zero curvature throughout."""
    x = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
    y = 2.0 * x + 1.0
    spline = CubicSpline(x, y, boundary="clamped", fpa=2.0, fpb=2.0)
    x_test = np.linspace(0.0, 5.0, 50)
    np.testing.assert_allclose(spline.evaluate(x_test), 2.0 * x_test + 1.0, atol=1e-8)


def test_clamped_spline_reproduces_a_cubic_exactly():
    """A cubic spline (piecewise cubic with matching values/slopes/2nd
    derivatives) interpolating a *global* cubic polynomial, clamped to
    that same cubic's exact end-slopes, must reproduce it exactly -- a
    single cubic already satisfies every continuity condition the spline
    enforces."""
    f = lambda x: x**3 - 2.0 * x**2 + x - 1.0
    fp = lambda x: 3.0 * x**2 - 4.0 * x + 1.0
    x = np.array([0.0, 0.5, 1.3, 2.0, 3.0])
    y = f(x)
    spline = CubicSpline(x, y, boundary="clamped", fpa=fp(x[0]), fpb=fp(x[-1]))
    x_test = np.linspace(0.0, 3.0, 60)
    np.testing.assert_allclose(spline.evaluate(x_test), f(x_test), atol=1e-8)


def test_rejects_non_increasing_x():
    with pytest.raises(ValueError):
        CubicSpline([0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0])


def test_clamped_requires_end_slopes():
    with pytest.raises(ValueError):
        CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], boundary="clamped")
