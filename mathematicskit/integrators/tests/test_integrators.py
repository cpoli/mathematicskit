"""Tests for mathematicskit.integrators against analytically-known ODE solutions."""

import numpy as np
import pytest
from scipy.optimize import fsolve

from mathematicskit._jit import njit
from mathematicskit.integrators import (
    BVPResult,
    collocation_bvp,
    dopri5_integrate,
    dopri5_step,
    implicit_euler_integrate,
    implicit_euler_step,
    leapfrog_integrate,
    leapfrog_step,
    rk4_integrate,
    rk4_step,
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


@njit
def _cos_rhs(state, t, params):
    """y' = cos t, exact solution y = sin t from y(0) = 0."""
    return np.cos(t) * np.ones_like(state)


@njit
def _minus_sin_force(pos, t, params):
    """pos'' = -sin t, exact solution pos = sin t from pos(0) = 0, vel(0) = 1."""
    return -np.sin(t) * np.ones_like(pos)


@njit
def _monomial_rhs(state, t, params):
    """y' = (p + 1) t**p, exact solution y = t**(p + 1) from y(0) = 0."""
    p = params[0]
    return (p + 1.0) * t**p * np.ones_like(state)


def test_rk4_is_fourth_order():
    params = np.array([1.0])
    errors = []
    for n in (20, 40):
        _, ys = rk4_integrate(_harmonic_rhs, np.array([1.0, 0.0]), 0.0, 1.0 / n, n, params)
        errors.append(abs(ys[-1, 0] - np.cos(1.0)))

    assert 12.0 < errors[0] / errors[1] < 20.0


def test_leapfrog_is_second_order():
    params = np.array([1.0])
    errors = []
    for n in (50, 100):
        _, positions, _ = leapfrog_integrate(_harmonic_force, np.array([1.0]), np.array([0.0]), 0.0, 1.0 / n, n, params)
        errors.append(abs(positions[-1, 0] - np.cos(1.0)))

    assert 3.5 < errors[0] / errors[1] < 4.5


def test_rk4_step_evaluates_stages_at_the_right_times():
    """For y' = cos t, one RK4 step is Simpson's rule on cos over [t, t + dt]."""
    t, dt = 0.3, 0.2
    y = rk4_step(_cos_rhs, np.array([0.0]), t, dt, np.zeros(1))
    simpson = dt / 6.0 * (np.cos(t) + 4.0 * np.cos(t + dt / 2) + np.cos(t + dt))
    assert y[0] == pytest.approx(simpson, rel=1e-14)


def test_rk4_and_dopri5_track_time_dependent_rhs():
    ts, ys = rk4_integrate(_cos_rhs, np.array([0.0]), 0.0, 1e-2, 500, np.zeros(1))
    np.testing.assert_allclose(ys[:, 0], np.sin(ts), atol=1e-9)

    ts, ys = dopri5_integrate(_cos_rhs, np.array([0.0]), 0.0, 5.0, 1e-2, np.zeros(1), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ys[:, 0], np.sin(ts), atol=1e-8)


def test_yoshida4_tracks_time_dependent_force():
    """The composition's negative middle sub-step must still pass the
    correct intermediate times to a time-dependent force."""
    ts, positions, velocities = yoshida4_integrate(_minus_sin_force, np.array([0.0]), np.array([1.0]), 0.0, 1e-2, 500, np.zeros(1))
    np.testing.assert_allclose(positions[:, 0], np.sin(ts), atol=1e-7)
    np.testing.assert_allclose(velocities[:, 0], np.cos(ts), atol=1e-7)


@pytest.mark.parametrize("step", [leapfrog_step, yoshida4_step])
def test_symplectic_steps_are_time_reversible(step):
    """Stepping forward by dt and then by -dt returns to the start (up to rounding)."""
    params = np.array([1.7])
    pos0, vel0 = np.array([0.8, -0.3]), np.array([0.1, 0.5])
    pos = pos0.copy()
    vel = vel0.copy()
    for _ in range(100):
        pos, vel = step(_harmonic_force, pos, vel, 0.0, 0.05, params)
    for _ in range(100):
        pos, vel = step(_harmonic_force, pos, vel, 0.0, -0.05, params)
    np.testing.assert_allclose(pos, pos0, atol=1e-12)
    np.testing.assert_allclose(vel, vel0, atol=1e-12)


def test_dopri5_step_is_exact_for_degree_five_polynomial_solution():
    """A 5th-order step integrates y' = 5 t**4 exactly, while the embedded
    4th-order solution does not, so the error estimate is nonzero."""
    y, err = dopri5_step(_monomial_rhs, np.array([0.0]), 0.0, 1.0, np.array([4.0]))
    assert y[0] == pytest.approx(1.0, abs=1e-14)
    assert abs(err[0]) > 1e-6


def test_dopri5_step_error_estimate_vanishes_for_degree_four_polynomial_solution():
    """Both embedded solutions are exact for y' = 4 t**3, so their difference is zero."""
    y, err = dopri5_step(_monomial_rhs, np.array([0.0]), 0.0, 1.0, np.array([3.0]))
    assert y[0] == pytest.approx(1.0, abs=1e-14)
    assert err[0] == pytest.approx(0.0, abs=1e-14)


def test_dopri5_grows_output_buffer_past_initial_capacity():
    """dt_max forces more accepted steps than the initial 1024-row buffer holds."""
    ts, ys = dopri5_integrate(_harmonic_rhs, np.array([1.0, 0.0]), 0.0, 3.0, 1e-3, np.array([2.0]), dt_max=1e-3)

    assert len(ts) > 2048
    assert ts[-1] == 3.0
    assert np.all(np.diff(ts) > 0.0)
    np.testing.assert_allclose(ys[:, 0], np.cos(2.0 * ts), atol=1e-9)


def test_dopri5_stops_early_when_max_steps_exhausted():
    ts, ys = dopri5_integrate(_harmonic_rhs, np.array([1.0, 0.0]), 0.0, 10.0, 1e-3, np.array([2.0]), dt_max=1e-2, max_steps=50)

    assert len(ts) == len(ys) <= 51
    assert ts[-1] < 10.0
    np.testing.assert_allclose(ys[:, 0], np.cos(2.0 * ts), atol=1e-6)


def test_dopri5_accepts_steps_at_dt_min_regardless_of_error():
    """With an unattainable tolerance, steps shrink to dt_min and are then
    accepted anyway, so integration still reaches t_end."""
    ts, _ = dopri5_integrate(_harmonic_rhs, np.array([1.0, 0.0]), 0.0, 1.0, 0.1, np.array([2.0]), rtol=0.0, atol=1e-20, dt_min=0.05)

    assert ts[-1] == 1.0
    np.testing.assert_allclose(np.diff(ts)[:-1], 0.05)


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


@njit
def _quadratic_decay_rhs(state, t, params):
    return -(state**2)


@njit
def _blowup_rhs(state, t, params):
    """y' = y**2 from y(0) = 1 has solution 1 / (1 - t), singular at t = 1."""
    return state**2


def test_implicit_euler_step_solves_nonlinear_equation():
    """For y' = -y**2 backward Euler solves dt y**2 + y - y0 = 0, whose
    positive root is (-1 + sqrt(1 + 4 dt y0)) / (2 dt)."""
    y0, dt = 3.0, 0.5
    y = implicit_euler_step(_quadratic_decay_rhs, np.array([y0]), 0.0, dt, np.zeros(1))
    assert y[0] == pytest.approx((-1.0 + np.sqrt(1.0 + 4.0 * dt * y0)) / (2.0 * dt), rel=1e-10)


def test_implicit_euler_step_raises_when_newton_does_not_converge():
    with pytest.raises(RuntimeError, match="did not converge"):
        implicit_euler_step(_quadratic_decay_rhs, np.array([3.0]), 0.0, 0.5, np.zeros(1), 1e-10, 1)


def test_stiff_integrate_raises_when_solver_fails():
    with pytest.raises(RuntimeError, match="stiff_integrate"):
        stiff_integrate(_blowup_rhs, [1.0], 0.0, 2.0, np.zeros(1))


def test_collocation_bvp_reports_failure_when_no_solution_exists():
    """Bratu's problem has no solution for lam above lam_c ~ 3.5138."""
    x = np.linspace(0.0, 1.0, 11)
    res = collocation_bvp(_bratu_rhs, _dirichlet_zero_bc, x, np.zeros((11, 2)), np.array([5.0]), max_nodes=200)

    assert not res.success
    assert res.message
