"""Tests for Verhulst's logistic growth and the Lotka-Volterra system."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.population import LogisticGrowth, LotkaVolterra, logistic_growth_solution, lotka_volterra_invariant
from mathematicskit.ode_dynamics.utils.period_estimation import estimate_period


def test_logistic_integration_matches_closed_form():
    system = LogisticGrowth([5.0], r=0.8, K=50.0)
    result = system.integrate((0.0, 15.0), dt=1e-2, method="rk4")
    exact = logistic_growth_solution(result.t, n0=5.0, r=0.8, K=50.0)
    np.testing.assert_allclose(result.y[:, 0], exact, rtol=1e-8)


def test_logistic_inflection_at_half_carrying_capacity():
    """N(t) passes K/2 at t* = ln((K - N0)/N0) / r, where growth is fastest."""
    n0, r, K = 1.0, 2.0, 101.0
    t_star = np.log((K - n0) / n0) / r
    assert logistic_growth_solution(t_star, n0, r, K) == pytest.approx(K / 2)


def test_logistic_rejects_nonpositive_n0():
    with pytest.raises(ValueError):
        logistic_growth_solution(1.0, n0=0.0, r=1.0, K=1.0)


def test_lotka_volterra_invariant_is_conserved():
    params = dict(alpha=1.1, beta=0.4, delta=0.1, gamma=0.4)
    system = LotkaVolterra([10.0, 10.0], **params)
    result = system.integrate((0.0, 50.0), dt=1e-3, method="rk4")
    V = lotka_volterra_invariant(result.y[:, 0], result.y[:, 1], **params)
    assert np.max(np.abs(V - V[0])) < 1e-8


def test_lotka_volterra_fixed_point_is_stationary_and_invariant_minimum():
    params = dict(alpha=1.1, beta=0.4, delta=0.1, gamma=0.4)
    system = LotkaVolterra([1.0, 1.0], **params)
    fp = system.fixed_point()
    np.testing.assert_allclose(system.rhs(fp), 0.0, atol=1e-14)
    V0 = lotka_volterra_invariant(*fp, **params)
    assert lotka_volterra_invariant(fp[0] * 1.1, fp[1], **params) > V0
    assert lotka_volterra_invariant(fp[0], fp[1] * 0.9, **params) > V0


def test_lotka_volterra_small_oscillation_period():
    """Linearizing about the center gives period 2*pi/sqrt(alpha*gamma)."""
    alpha, gamma = 2.0, 0.5
    system = LotkaVolterra([0.505, 4.0], alpha=alpha, beta=0.5, delta=1.0, gamma=gamma)
    result = system.integrate((0.0, 100.0), dt=1e-3, method="rk4")
    period = estimate_period(result.t, result.y[:, 0] - system.fixed_point()[0])
    assert period == pytest.approx(2.0 * np.pi / np.sqrt(alpha * gamma), rel=1e-3)
