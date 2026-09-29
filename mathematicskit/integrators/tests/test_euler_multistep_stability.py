"""Tests for explicit Euler, Adams-Bashforth, and the linear stability
regions, against closed forms (Hairer, Nørsett & Wanner, Solving ODEs I,
sec. III.1; Hairer & Wanner, Solving ODEs II, sec. IV.2 and V.1)."""

import numpy as np
import pytest

from mathematicskit._jit import njit
from mathematicskit.integrators import (
    ADAMS_BASHFORTH_COEFFICIENTS,
    STABILITY_METHODS,
    adams_bashforth_boundary_locus,
    adams_bashforth_integrate,
    euler_integrate,
    euler_step,
    is_absolutely_stable,
    rk4_step,
    stability_function,
)
from mathematicskit.pde import RK4_REAL_AXIS_LIMIT


@njit
def _linear_rhs(state, t, params):
    return params[0] * state


@njit
def _quadratic_in_t_rhs(state, t, params):
    """y' = 3 t**2, exact solution y = t**3 from y(0) = 0."""
    return 3.0 * t**2 * np.ones_like(state)


def test_euler_step_is_the_tangent_line():
    y = euler_step(_linear_rhs, np.array([2.0, -1.0]), 0.0, 0.1, np.array([-3.0]))
    np.testing.assert_allclose(y, np.array([2.0, -1.0]) * (1.0 - 0.3), rtol=1e-15)


def test_euler_integrate_matches_closed_form_recurrence():
    """On y' = lam y, n Euler steps give exactly (1 + lam h)**n."""
    ts, ys = euler_integrate(_linear_rhs, np.array([1.0]), 0.5, 0.1, 10, np.array([-1.0]))
    np.testing.assert_allclose(ys[:, 0], 0.9 ** np.arange(11), rtol=1e-13)
    np.testing.assert_allclose(ts, 0.5 + 0.1 * np.arange(11))


def test_euler_is_first_order():
    errors = []
    for n in (50, 100):
        _, ys = euler_integrate(_linear_rhs, np.array([1.0]), 0.0, 1.0 / n, n, np.array([-1.0]))
        errors.append(abs(ys[-1, 0] - np.exp(-1.0)))
    assert 1.9 < errors[0] / errors[1] < 2.1


def test_adams_bashforth_order_one_is_euler():
    args = (_linear_rhs, np.array([1.0, 2.0]), 0.0, 0.05, 20, np.array([-2.0]))
    np.testing.assert_allclose(adams_bashforth_integrate(*args, 1)[1], euler_integrate(*args)[1], rtol=1e-15)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_adams_bashforth_converges_at_its_order(order):
    errors = []
    for n in (40, 80):
        _, ys = adams_bashforth_integrate(_linear_rhs, np.array([1.0]), 0.0, 1.0 / n, n, np.array([-1.0]), order)
        errors.append(abs(ys[-1, 0] - np.exp(-1.0)))
    ratio = errors[0] / errors[1]
    assert 0.9 * 2**order < ratio < 1.1 * 2**order


def test_adams_bashforth_starts_with_rk4_steps():
    ts, ys = adams_bashforth_integrate(_linear_rhs, np.array([1.0]), 0.0, 0.1, 5, np.array([-1.0]), 4)
    y = np.array([1.0])
    for i in range(3):
        y = rk4_step(_linear_rhs, y, ts[i], 0.1, np.array([-1.0]))
        np.testing.assert_allclose(ys[i + 1], y, rtol=1e-15)


def test_adams_bashforth_step_uses_the_tabulated_weights():
    """Step 4 of AB4 is y_3 + h (55 f_3 - 59 f_2 + 37 f_1 - 9 f_0) / 24."""
    lam, h = -0.7, 0.1
    ts, ys = adams_bashforth_integrate(_linear_rhs, np.array([1.0]), 0.0, h, 4, np.array([lam]), 4)
    slopes = lam * ys[:4, 0][::-1]
    expected = ys[3, 0] + h * ADAMS_BASHFORTH_COEFFICIENTS[3] @ slopes
    assert ys[4, 0] == pytest.approx(expected, rel=1e-15)


def test_adams_bashforth_three_is_exact_for_quadratic_slopes():
    """AB-k integrates y' = p(t) exactly when deg p < k (it integrates the
    interpolating polynomial of the slopes)."""
    ts, ys = adams_bashforth_integrate(_quadratic_in_t_rhs, np.array([0.0]), 0.0, 0.1, 30, np.zeros(1), 3)
    np.testing.assert_allclose(ys[:, 0], ts**3, atol=1e-12)


@pytest.mark.parametrize("order", [0, 5])
def test_adams_bashforth_rejects_unsupported_order(order):
    with pytest.raises(ValueError, match="order"):
        adams_bashforth_integrate(_linear_rhs, np.array([1.0]), 0.0, 0.1, 3, np.array([-1.0]), order)


def test_stability_functions_have_closed_forms():
    z = np.array([-0.5, 0.3 + 0.4j, -2.0 - 1.0j])
    np.testing.assert_allclose(stability_function("euler", z), 1.0 + z)
    np.testing.assert_allclose(stability_function("implicit_euler", z), 1.0 / (1.0 - z))
    np.testing.assert_allclose(stability_function("rk4", z), sum(z**k / np.prod(np.arange(1, k + 1)) for k in range(5)))
    with pytest.raises(ValueError):
        stability_function("adams_bashforth2", z)


def test_stability_function_is_the_one_step_amplification_factor():
    lam, h = -1.3, 0.4
    y = rk4_step(_linear_rhs, np.array([1.0]), 0.0, h, np.array([lam]))
    assert y[0] == pytest.approx(stability_function("rk4", lam * h).real, rel=1e-14)


@pytest.mark.parametrize(
    "method,left_end",
    [
        ("euler", -2.0),
        ("rk4", -RK4_REAL_AXIS_LIMIT),
        ("adams_bashforth2", -1.0),
        ("adams_bashforth3", -6.0 / 11.0),
        ("adams_bashforth4", -3.0 / 10.0),
    ],
)
def test_real_stability_intervals(method, left_end):
    assert is_absolutely_stable(method, left_end * 0.99)
    assert not is_absolutely_stable(method, left_end * 1.01)
    assert not is_absolutely_stable(method, 0.01)


def test_implicit_euler_is_a_stable():
    """Dahlquist: the whole left half-plane lies in backward Euler's region."""
    x, y = np.meshgrid(np.linspace(-1e3, 0.0, 41), np.linspace(-1e3, 1e3, 41))
    assert is_absolutely_stable("implicit_euler", x + 1j * y).all()
    assert not is_absolutely_stable("implicit_euler", 0.5)


def test_stability_region_predicts_adams_bashforth_growth():
    """Inside AB2's region a decaying mode decays; just outside it grows."""
    h = 0.1
    for z, stable in ((-0.9, True), (-1.1, False)):
        assert bool(is_absolutely_stable("adams_bashforth2", z)) is stable
        _, ys = adams_bashforth_integrate(_linear_rhs, np.array([1.0]), 0.0, h, 400, np.array([z / h]), 2)
        assert bool(abs(ys[-1, 0]) < 1.0) is stable


def test_is_absolutely_stable_keeps_shape_and_rejects_unknown_methods():
    z = np.zeros((3, 4), dtype=complex) - 0.1
    for method in STABILITY_METHODS:
        mask = is_absolutely_stable(method, z)
        assert mask.shape == (3, 4)
        assert mask.all()
    with pytest.raises(ValueError):
        is_absolutely_stable("leapfrog", z)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_boundary_locus_puts_a_characteristic_root_on_the_unit_circle(order):
    beta = ADAMS_BASHFORTH_COEFFICIENTS[order - 1, :order]
    for z in adams_bashforth_boundary_locus(order, n_points=13)[1:-1]:
        # rho(zeta) - z sigma(zeta), highest power first
        coeffs = np.zeros(order + 1, dtype=complex)
        coeffs[0], coeffs[1] = 1.0, -1.0
        coeffs[1:] -= z * beta
        assert np.min(np.abs(np.abs(np.roots(coeffs)) - 1.0)) < 1e-9


def test_boundary_locus_rejects_unsupported_order():
    with pytest.raises(ValueError):
        adams_bashforth_boundary_locus(5)
