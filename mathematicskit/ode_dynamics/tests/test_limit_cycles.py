"""Tests for the Van der Pol oscillator's limit cycle."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.limit_cycles import VanDerPolOscillator, bendixson_criterion, estimate_limit_cycle_amplitude
from mathematicskit.ode_dynamics.utils.period_estimation import estimate_period


def test_all_nonzero_initial_conditions_converge_to_same_amplitude():
    """The hallmark of a limit cycle (vs. a center): trajectories from
    very different initial conditions converge to the *same* amplitude."""
    amp_small = estimate_limit_cycle_amplitude(mu=1.0, t_transient=100.0, t_observe=30.0)

    system_large = VanDerPolOscillator([5.0, 0.0], mu=1.0)
    system_large.integrate((0.0, 100.0), dt=1e-3, method="rk4")
    result_large = system_large.integrate((system_large.t, system_large.t + 30.0), dt=1e-3, method="rk4")
    amp_large = np.max(np.abs(result_large.y[:, 0]))

    assert abs(amp_small - amp_large) < 0.05


def test_amplitude_is_near_two_for_mu_equals_one():
    amp = estimate_limit_cycle_amplitude(mu=1.0, t_transient=100.0, t_observe=30.0)
    assert 1.8 < amp < 2.2


def test_zero_mu_reduces_to_harmonic_oscillator_energy_conservation():
    system = VanDerPolOscillator([1.0, 0.0], mu=0.0)
    result = system.integrate((0.0, 20.0), dt=1e-3, method="rk4")
    energy = 0.5 * (result.y[:, 0] ** 2 + result.y[:, 1] ** 2)
    assert np.max(np.abs(energy - energy[0])) < 1e-2


def test_period_estimation_matches_known_sine_period():
    t = np.linspace(0.0, 20.0, 20000)
    x = np.sin(2.0 * np.pi * t / 3.0)
    assert estimate_period(t, x) == pytest.approx(3.0, abs=0.02)


def test_bendixson_divergence_matches_closed_form():
    res = bendixson_criterion(lambda x, y: (y, 2.0 * (1.0 - x**2) * y - x), (-2, 2), (-2, 2), n=21)
    np.testing.assert_allclose(res.divergence, 2.0 * (1.0 - res.X**2), atol=1e-6)
    assert not res.rules_out_periodic_orbits


def test_bendixson_rules_out_cycles_for_dissipative_flows_but_not_a_center():
    assert bendixson_criterion(lambda x, y: (y - x**3, -x - y**3 - y), (-2, 2), (-2, 2)).rules_out_periodic_orbits
    assert bendixson_criterion(lambda x, y: (y - x, -x - y), (-2, 2), (-2, 2)).rules_out_periodic_orbits
    assert not bendixson_criterion(lambda x, y: (y, -x), (-2, 2), (-2, 2)).rules_out_periodic_orbits
