"""Tests for finite-difference derivatives against closed-form derivatives."""

import numpy as np
import pytest

from mathematicskit.calculus.systems.finite_differences import backward_difference, central_difference, forward_difference, richardson_extrapolation


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


def test_complex_step_is_exact_to_machine_precision():
    import numpy as np

    from mathematicskit.calculus.systems.finite_differences import complex_step_derivative

    for x in (0.3, 1.0, 2.5):
        assert complex_step_derivative(np.sin, x) == pytest.approx(np.cos(x), abs=1e-15)
        assert complex_step_derivative(lambda t: np.exp(t) / np.sqrt(t), x) == pytest.approx(np.exp(x) / np.sqrt(x) * (1 - 0.5 / x), rel=1e-14)


def test_complex_step_has_no_cancellation_at_tiny_steps():
    import numpy as np

    from mathematicskit.calculus.systems.finite_differences import central_difference, complex_step_derivative

    exact = np.cos(1.0)
    assert abs(complex_step_derivative(np.sin, 1.0, h=1e-100) - exact) < 1e-15
    assert abs(central_difference(np.sin, 1.0, h=1e-12) - exact) > 1e-6  # cancellation ruins the difference
