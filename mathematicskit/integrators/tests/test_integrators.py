"""Tests for mathematicskit.integrators against analytically-known ODE solutions."""

import numpy as np
import pytest
from scipy.optimize import fsolve

from mathematicskit._jit import njit
from mathematicskit.integrators import (
    BVPResult,
    collocation_bvp,
    dopri5_integrate,
    implicit_euler_integrate,
    implicit_euler_step,
    leapfrog_integrate,
    leapfrog_step,
    rk4_integrate,
    stiff_integrate,
    velocity_verlet_integrate,
    velocity_verlet_step,
    yoshida4_integrate,
    yoshida4_step,
)


@njit
def _harmonic_force(pos, t, params):
    omega = params[0]
    return -(omega**2) * pos


@njit
def _harmonic_rhs(state, t, params):
    omega = params[0]
    out = np.empty(2)
    out[0] = state[1]
    out[1] = -(omega**2) * state[0]
    return out


@njit
def _zero_rhs(state, t, params):
    return np.zeros_like(state)


def _harmonic_energy(pos, vel, omega):
    return 0.5 * vel**2 + 0.5 * omega**2 * pos**2


def test_velocity_verlet_is_an_alias_for_leapfrog():
    """The two names refer to the same kick-drift-kick scheme, so they
    must be the identical function object, not merely numerically
    equivalent copies."""
    assert velocity_verlet_step is leapfrog_step
    assert velocity_verlet_integrate is leapfrog_integrate


def test_dopri5_matches_analytic_harmonic_oscillator():
    omega = 2.0
    params = np.array([omega])
    state0 = np.array([1.0, 0.0])
    t_end = 10.0

    ts, ys = dopri5_integrate(_harmonic_rhs, state0, 0.0, t_end, 0.01, params, rtol=1e-10, atol=1e-12)

    assert ts[0] == 0.0
    assert ts[-1] == t_end
    expected = np.cos(omega * ts)
    np.testing.assert_allclose(ys[:, 0], expected, atol=1e-6)


def test_dopri5_energy_drift_shrinks_with_tolerance():
    omega = 2.0
    params = np.array([omega])
    state0 = np.array([1.0, 0.0])
    t_end = 20.0

    drifts = []
    for rtol in (1e-4, 1e-9):
        _, ys = dopri5_integrate(_harmonic_rhs, state0, 0.0, t_end, 0.01, params, rtol=rtol, atol=rtol * 1e-3)
        energy = _harmonic_energy(ys[:, 0], ys[:, 1], omega)
        drifts.append(np.max(np.abs(energy - energy[0]) / energy[0]))

    assert drifts[1] < drifts[0]


def test_dopri5_takes_fewer_steps_for_looser_tolerance():
    omega = 2.0
    params = np.array([omega])
    state0 = np.array([1.0, 0.0])
    t_end = 20.0

    ts_tight, _ = dopri5_integrate(_harmonic_rhs, state0, 0.0, t_end, 0.01, params, rtol=1e-10, atol=1e-12)
    ts_loose, _ = dopri5_integrate(_harmonic_rhs, state0, 0.0, t_end, 0.01, params, rtol=1e-3, atol=1e-6)

    assert len(ts_loose) < len(ts_tight)


def test_dopri5_agrees_with_fixed_step_rk4_at_tight_tolerance():
    omega = 2.0
    params = np.array([omega])
    state0 = np.array([1.0, 0.0])
    t_end = 5.0

    ts, ys = dopri5_integrate(_harmonic_rhs, state0, 0.0, t_end, 0.01, params, rtol=1e-10, atol=1e-12)
    _, ys_rk4 = rk4_integrate(_harmonic_rhs, state0, 0.0, 1e-4, 50000, params)

    np.testing.assert_allclose(ys[-1], ys_rk4[-1], atol=1e-6)


def test_rk4_matches_analytic_harmonic_oscillator():
    omega = 3.0
    params = np.array([omega])
    state0 = np.array([1.0, 0.0])
    ts, ys = rk4_integrate(_harmonic_rhs, state0, 0.0, 1e-3, 2000, params)
    expected = np.cos(omega * ts)
    np.testing.assert_allclose(ys[:, 0], expected, atol=1e-5)


def test_leapfrog_conserves_energy_of_harmonic_oscillator():
    omega = 2.0
    params = np.array([omega])
    pos0 = np.array([1.0])
    vel0 = np.array([0.0])
    pos, vel = pos0.copy(), vel0.copy()
    e0 = _harmonic_energy(pos[0], vel[0], omega)
    t = 0.0
    dt = 1e-3
    for _ in range(20000):
        pos, vel = leapfrog_step(_harmonic_force, pos, vel, t, dt, params)
        t += dt
    e1 = _harmonic_energy(pos[0], vel[0], omega)
    assert abs(e1 - e0) / e0 < 1e-4


def test_leapfrog_integrate_matches_repeated_leapfrog_step():
    params = np.array([2.0])
    pos0, vel0 = np.array([1.0]), np.array([0.0])
    ts, positions, velocities = leapfrog_integrate(_harmonic_force, pos0, vel0, 0.5, 1e-2, 100, params)

    assert ts.shape == (101,)
    assert positions.shape == velocities.shape == (101, 1)
    np.testing.assert_allclose(ts, 0.5 + 1e-2 * np.arange(101))
    pos, vel = pos0.copy(), vel0.copy()
    for i in range(100):
        pos, vel = leapfrog_step(_harmonic_force, pos, vel, ts[i], 1e-2, params)
    np.testing.assert_allclose(positions[-1], pos)
    np.testing.assert_allclose(velocities[-1], vel)


def test_yoshida4_integrate_matches_analytic_harmonic_oscillator():
    omega = 2.0
    params = np.array([omega])
    ts, positions, velocities = yoshida4_integrate(_harmonic_force, np.array([1.0]), np.array([0.0]), 0.0, 1e-2, 1000, params)

    assert ts.shape == (1001,)
    np.testing.assert_allclose(positions[:, 0], np.cos(omega * ts), atol=1e-6)
    np.testing.assert_allclose(velocities[:, 0], -omega * np.sin(omega * ts), atol=1e-6)


def test_yoshida4_step_is_fourth_order():
    """Halving dt should cut the global error by ~2**4 = 16 (vs ~4 for
    leapfrog), confirming the Yoshida composition raises the order."""
    omega = 1.0
    params = np.array([omega])
    t_end = 1.0
    errors = []
    for n in (20, 40):
        dt = t_end / n
        pos, vel = np.array([1.0]), np.array([0.0])
        for i in range(n):
            pos, vel = yoshida4_step(_harmonic_force, pos, vel, i * dt, dt, params)
        errors.append(abs(pos[0] - np.cos(omega * t_end)))

    assert 12.0 < errors[0] / errors[1] < 20.0


def test_dopri5_zero_error_grows_step_to_dt_max():
    """With an identically-zero RHS every step's error estimate is exactly
    zero, so the controller grows dt by its maximum factor until capped."""
    state0 = np.array([1.0, -2.0])
    ts, ys = dopri5_integrate(_zero_rhs, state0, 0.0, 10.0, 1e-3, np.zeros(1), dt_max=1.0)

    np.testing.assert_allclose(ys, np.tile(state0, (len(ts), 1)))
    assert ts[-1] == 10.0
    np.testing.assert_allclose(np.diff(ts)[-5:-1], 1.0)  # final step is clipped to land on t_end


# --- Stiff (implicit) integrators and boundary-value problems ------------------


@njit
def _linear_decay_rhs(state, t, params):
    return -params[0] * state


@njit
def _prothero_robinson_rhs(state, t, params):
    """y' = -lam (y - cos t) - sin t, exact solution y = cos t from y(0) = 1."""
    lam = params[0]
    return -lam * (state - np.cos(t)) - np.sin(t)


def test_implicit_euler_step_matches_closed_form_for_linear_decay():
    """For y' = -lam y backward Euler is y_new = y / (1 + lam dt) exactly."""
    lam, dt = 50.0, 0.1
    y = implicit_euler_step(_linear_decay_rhs, np.array([2.0, -1.0]), 0.0, dt, np.array([lam]))
    np.testing.assert_allclose(y, np.array([2.0, -1.0]) / (1.0 + lam * dt), rtol=1e-12)


def test_implicit_euler_is_first_order():
    params = np.array([1.0])
    errors = []
    for n in (50, 100):
        _, ys = implicit_euler_integrate(_linear_decay_rhs, np.array([1.0]), 0.0, 1.0 / n, n, params)
        errors.append(abs(ys[-1, 0] - np.exp(-1.0)))

    assert 1.8 < errors[0] / errors[1] < 2.2


def test_implicit_euler_stable_where_rk4_blows_up_on_stiff_problem():
    """lam * dt = 100 is far outside RK4's stability region (~2.785) but
    inside backward Euler's (the whole left half-plane)."""
    params = np.array([1e4])
    dt, n = 1e-2, 300
    ts, ys_ie = implicit_euler_integrate(_prothero_robinson_rhs, np.array([1.0]), 0.0, dt, n, params)
    _, ys_rk4 = rk4_integrate(_prothero_robinson_rhs, np.array([1.0]), 0.0, dt, n, params)

    np.testing.assert_allclose(ys_ie[:, 0], np.cos(ts), atol=1e-5)
    assert not np.all(np.isfinite(ys_rk4)) or np.max(np.abs(ys_rk4)) > 1e6


@pytest.mark.parametrize("method", ["Radau", "BDF", "LSODA"])
def test_stiff_integrate_matches_analytic_solution(method):
    ts, ys = stiff_integrate(_prothero_robinson_rhs, [1.0], 0.0, 3.0, np.array([1e4]), method=method, rtol=1e-8, atol=1e-10)

    assert ts[0] == 0.0 and ts[-1] == 3.0
    assert ys.shape == (ts.size, 1)
    np.testing.assert_allclose(ys[:, 0], np.cos(ts), atol=1e-6)


def test_stiff_integrate_takes_far_fewer_steps_than_dopri5():
    params = np.array([1e4])
    ts_radau, _ = stiff_integrate(_prothero_robinson_rhs, [1.0], 0.0, 3.0, params, rtol=1e-6, atol=1e-9)
    ts_dp, _ = dopri5_integrate(_prothero_robinson_rhs, np.array([1.0]), 0.0, 3.0, 1e-3, params, rtol=1e-6, atol=1e-9)

    assert 10 * len(ts_radau) < len(ts_dp)


def test_stiff_integrate_honours_t_eval_and_jacobian():
    params = np.array([1e3])
    t_eval = np.linspace(0.0, 1.0, 5)

    def jac(state, t, p):
        return np.array([[-p[0]]])

    ts, ys = stiff_integrate(_linear_decay_rhs, [1.0], 0.0, 1.0, params, method="BDF", t_eval=t_eval, jac=jac)
    np.testing.assert_array_equal(ts, t_eval)
    assert ys.shape == (5, 1)


def test_stiff_integrate_rejects_explicit_method():
    with pytest.raises(ValueError, match="method"):
        stiff_integrate(_linear_decay_rhs, [1.0], 0.0, 1.0, np.array([1.0]), method="RK45")


@njit
def _oscillator_bvp_rhs(state, x, params):
    out = np.empty(2)
    out[0] = state[1]
    out[1] = -state[0]
    return out


@njit
def _bratu_rhs(state, x, params):
    """Bratu's problem y'' + lam e^y = 0."""
    out = np.empty(2)
    out[0] = state[1]
    out[1] = -params[0] * np.exp(state[0])
    return out


def _dirichlet_zero_bc(ya, yb, params):
    return np.array([ya[0], yb[0]])


def test_collocation_bvp_recovers_sine():
    x = np.linspace(0.0, np.pi / 2, 5)
    res = collocation_bvp(_oscillator_bvp_rhs, lambda ya, yb, p: np.array([ya[0], yb[0] - 1.0]), x, np.zeros((5, 2)), np.zeros(1), tol=1e-8)

    assert isinstance(res, BVPResult) and res.success
    assert res.y.shape == (res.x.size, 2)
    xq = np.linspace(0.0, np.pi / 2, 50)
    np.testing.assert_allclose(res.sol(xq)[:, 0], np.sin(xq), atol=1e-7)
    np.testing.assert_allclose(res.sol(xq)[:, 1], np.cos(xq), atol=1e-6)


def test_collocation_bvp_finds_both_bratu_branches_from_different_guesses():
    """Bratu's problem on [0, 1] with lam = 1 has two solutions,
    y = -2 ln(cosh((x - 1/2) theta / 2) / cosh(theta / 4)) with
    theta = sqrt(2 lam) cosh(theta / 4); the guess selects the branch."""
    params = np.array([1.0])
    x = np.linspace(0.0, 1.0, 11)
    low = collocation_bvp(_bratu_rhs, _dirichlet_zero_bc, x, np.zeros((11, 2)), params, tol=1e-8)
    guess_high = np.column_stack([4.0 * np.sin(np.pi * x), 4.0 * np.pi * np.cos(np.pi * x)])
    high = collocation_bvp(_bratu_rhs, _dirichlet_zero_bc, x, guess_high, params, tol=1e-8)

    assert low.success and high.success
    peaks = []
    for theta0 in (1.5, 10.0):
        theta = fsolve(lambda th: th - np.sqrt(2.0) * np.cosh(th / 4.0), theta0)[0]
        peaks.append(-2.0 * np.log(1.0 / np.cosh(theta / 4.0)))
    assert low.sol([0.5])[0, 0] == pytest.approx(peaks[0], abs=1e-6)
    assert high.sol([0.5])[0, 0] == pytest.approx(peaks[1], abs=1e-6)
