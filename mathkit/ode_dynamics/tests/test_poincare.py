"""Tests for the Duffing oscillator and its stroboscopic Poincare section."""

import numpy as np

from mathkit.ode_dynamics.systems.poincare import DuffingOscillator, stroboscopic_poincare_section


def test_duffing_integrates_without_error_and_has_expected_shape():
    system = DuffingOscillator([1.0, 0.0], delta=0.3, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2)
    result = system.integrate((0.0, 20.0), dt=1e-3, method="rk4")
    assert result.y.shape[1] == 2
    assert result.y.shape[0] == result.t.shape[0]


def test_undriven_undamped_duffing_conserves_energy():
    """With gamma=0 (no drive) and delta=0 (no damping), the Duffing
    system is the conservative double-well oscillator: total mechanical
    energy E = y^2/2 + alpha*x^2/2 + beta*x^4/4 should be conserved."""
    alpha, beta = -1.0, 1.0
    system = DuffingOscillator([0.5, 0.0], delta=0.0, alpha=alpha, beta=beta, gamma=0.0, omega=1.2)
    result = system.integrate((0.0, 20.0), dt=1e-3, method="rk4")
    x, y = result.y[:, 0], result.y[:, 1]
    energy = 0.5 * y**2 + 0.5 * alpha * x**2 + 0.25 * beta * x**4
    assert np.max(np.abs(energy - energy[0])) < 1e-3


def test_stroboscopic_section_returns_requested_number_of_points():
    system = DuffingOscillator([1.0, 0.0], delta=0.3, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2)
    xs, ys = stroboscopic_poincare_section(system, n_periods=8, n_transient_periods=4, dt=1e-2)
    assert xs.shape == (8,)
    assert ys.shape == (8,)
    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(ys))


def test_stroboscopic_section_of_a_periodic_orbit_converges_to_a_point():
    """For small enough drive amplitude, the long-time attractor is a
    single periodic orbit synchronized with the drive, so the
    stroboscopic section should converge to (nearly) one fixed point."""
    system = DuffingOscillator([0.1, 0.0], delta=0.5, alpha=1.0, beta=0.0, gamma=0.05, omega=1.0)
    xs, ys = stroboscopic_poincare_section(system, n_periods=10, n_transient_periods=30, dt=1e-2)
    assert np.std(xs) < 0.05
    assert np.std(ys) < 0.05
