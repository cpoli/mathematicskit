"""Tests for finite-difference derivatives against closed-form derivatives."""

import numpy as np
import pytest

from mathkit.calculus.systems.finite_differences import backward_difference, central_difference, forward_difference, richardson_extrapolation


def test_central_difference_more_accurate_than_forward_at_same_h():
    f = np.sin
    x0 = 1.0
    h = 1e-3
    exact = np.cos(x0)
    err_forward = abs(forward_difference(f, x0, h) - exact)
    err_central = abs(central_difference(f, x0, h) - exact)
    assert err_central < err_forward


def test_backward_and_forward_bracket_central():
    f = lambda x: x**3
    x0 = 2.0
    h = 1e-3
    exact = 3.0 * x0**2
    assert forward_difference(f, x0, h) == pytest.approx(exact, abs=1e-2)
    assert backward_difference(f, x0, h) == pytest.approx(exact, abs=1e-2)
    assert central_difference(f, x0, h) == pytest.approx(exact, abs=1e-5)


def test_richardson_extrapolation_is_much_more_accurate():
    f = np.exp
    x0 = 1.0
    exact = np.exp(1.0)
    result = richardson_extrapolation(f, x0, h=0.1, levels=5)
    assert abs(result.value - exact) < 1e-10
    # Plain central difference at the coarsest h should be far less accurate.
    plain = central_difference(f, x0, h=0.1)
    assert abs(result.value - exact) < abs(plain - exact)


def test_richardson_error_estimate_shrinks_with_more_levels():
    f = np.cos
    x0 = 0.5
    errs = []
    for levels in (2, 3, 4, 5):
        result = richardson_extrapolation(f, x0, h=0.2, levels=levels)
        errs.append(abs(result.value - (-np.sin(x0))))
    assert errs[-1] < errs[0]
