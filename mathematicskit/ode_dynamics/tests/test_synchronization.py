"""Tests for the Kuramoto model against Kuramoto's Lorentzian solution."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.synchronization import KuramotoModel, kuramoto_lorentzian_order_parameter, kuramoto_order_parameter


def _lorentzian_population(n, gamma):
    """Deterministic Lorentzian quantiles, and random initial phases."""
    omegas = gamma * np.tan(np.pi * (np.arange(n) + 0.5) / n - np.pi / 2)
    theta0 = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, n)
    return theta0, omegas


def test_rhs_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, 7)
    omegas = rng.normal(size=7)
    system = KuramotoModel(theta, omegas, K=1.3)
    pairwise = omegas + 1.3 / 7 * np.sin(theta[None, :] - theta[:, None]).sum(axis=1)
    np.testing.assert_allclose(system.rhs(theta), pairwise, atol=1e-12)
    np.testing.assert_allclose(system._rhs_njit(theta, 0.0, omegas), pairwise, atol=1e-12)


@pytest.mark.parametrize("K", [1.5, 3.0])
def test_steady_coherence_matches_lorentzian_theory(K):
    gamma = 0.5
    theta0, omegas = _lorentzian_population(1000, gamma)
    result = KuramotoModel(theta0, omegas, K=K).integrate((0.0, 60.0), dt=0.02, method="rk4")
    r, _ = kuramoto_order_parameter(result.y[-500:])
    assert r.mean() == pytest.approx(kuramoto_lorentzian_order_parameter(K, gamma), abs=0.03)


def test_incoherent_below_critical_coupling():
    theta0, omegas = _lorentzian_population(1000, 0.5)
    result = KuramotoModel(theta0, omegas, K=0.5).integrate((0.0, 60.0), dt=0.02, method="rk4")
    r, _ = kuramoto_order_parameter(result.y[-500:])
    assert r.mean() < 0.15


def test_lorentzian_order_parameter_closed_form():
    np.testing.assert_allclose(kuramoto_lorentzian_order_parameter([1.0, 2.0, 8.0], gamma=1.0), [0.0, 0.0, np.sqrt(0.75)])


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        KuramotoModel([0.0, 1.0], [0.0], K=1.0)
