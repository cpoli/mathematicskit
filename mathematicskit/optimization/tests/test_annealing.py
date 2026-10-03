"""Tests for simulated annealing."""

import numpy as np
import pytest
from scipy import optimize

from mathematicskit.optimization.systems.annealing import simulated_annealing


def rastrigin(x):
    return 10 * x.size + float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_escapes_local_minima_of_rastrigin_and_agrees_with_dual_annealing():
    result = simulated_annealing(rastrigin, [4.5, 4.5], step_size=0.5, t0=20.0, cooling=0.9995, n_iter=30000, seed=0)
    reference = optimize.dual_annealing(rastrigin, bounds=[(-5.12, 5.12)] * 2, seed=0)
    assert result.fun == pytest.approx(reference.fun, abs=1e-2)
    assert np.allclose(result.x, 0.0, atol=0.01)
    assert result.method == "simulated_annealing"


def test_zero_temperature_is_greedy_descent():
    # At T = 0 uphill moves are never accepted, so the energy never rises.
    result = simulated_annealing(rastrigin, [2.2, -1.1], step_size=0.1, cooling=lambda k: 0.0, n_iter=2000, seed=1)
    assert np.all(np.diff(result.extra["energies"]) <= 0)


def test_geometric_schedule_and_bookkeeping():
    result = simulated_annealing(lambda x: float(x @ x), [1.0], t0=2.0, cooling=0.5, n_iter=5, seed=2)
    assert np.allclose(result.extra["temperatures"], 2.0 * 0.5 ** np.arange(5))
    assert result.path.shape == (6, 1) and result.extra["energies"].shape == (6,)
    assert result.fun == pytest.approx(float(result.x @ result.x))
    assert result.fun <= result.extra["energies"].min() + 1e-15


def test_high_temperature_accepts_almost_everything():
    result = simulated_annealing(lambda x: float(x @ x), [0.0], step_size=0.1, t0=1e6, cooling=1.0, n_iter=2000, seed=3)
    assert result.extra["acceptance_rate"] > 0.99


def test_custom_neighbor_on_discrete_state():
    # Minimize the number of inversions of a permutation by random adjacent swaps.
    def inversions(p):
        return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])

    def swap(p, rng):
        q = list(p)
        i = int(rng.integers(len(q) - 1))
        q[i], q[i + 1] = q[i + 1], q[i]
        return q

    result = simulated_annealing(inversions, [5, 3, 0, 4, 1, 2], neighbor=swap, t0=2.0, cooling=0.995, n_iter=3000, seed=4)
    assert result.x == [0, 1, 2, 3, 4, 5] and result.fun == 0
    assert result.path.size == 0
