"""Tests for Galton-Watson branching, random walks, Brownian motion,
Erlang's loss formula, and continuous-time Markov chains."""

import math

import numpy as np
import pytest

from mathematicskit.probability.systems.branching import galton_watson_extinction_probability, galton_watson_simulate
from mathematicskit.probability.systems.continuous_time_markov import ctmc_stationary_distribution, ctmc_transition_matrix
from mathematicskit.probability.systems.queueing import erlang_b
from mathematicskit.probability.systems.stochastic_processes import (
    brownian_motion,
    random_walk_return_fraction,
    return_probability_1d,
    simple_random_walk,
)


def test_extinction_probability_quadratic_closed_form():
    # G(s) = p0 + p1 s + p2 s^2 = s has roots 1 and p0 / p2.
    assert galton_watson_extinction_probability([0.2, 0.3, 0.5]) == pytest.approx(0.4, abs=1e-8)
    assert galton_watson_extinction_probability([0.25, 0.25, 0.5]) == pytest.approx(0.5, abs=1e-8)


def test_extinction_probability_geometric_offspring():
    # p_j = (1-a) a^j has G(s) = (1-a)/(1-a s); the fixed point is (1-a)/a for a > 1/2.
    a = 0.7
    pmf = (1 - a) * a ** np.arange(200)
    pmf /= pmf.sum()
    assert galton_watson_extinction_probability(pmf) == pytest.approx((1 - a) / a, abs=1e-7)


def test_extinction_is_certain_when_not_supercritical():
    assert galton_watson_extinction_probability([0.5, 0.0, 0.5]) == 1.0
    assert galton_watson_extinction_probability([0.6, 0.3, 0.1]) == 1.0
    assert galton_watson_extinction_probability([0.0, 1.0]) == 0.0


def test_galton_watson_simulation_mean_growth_and_extinction():
    pmf = [0.2, 0.3, 0.5]  # mean 1.3
    result = galton_watson_simulate(pmf, n_generations=8, n_runs=20000, seed=3)
    assert result.generation_sizes.shape == (20000, 9)
    assert result.generation_sizes[:, 5].mean() == pytest.approx(1.3**5, rel=0.05)
    assert result.extinct_fraction == pytest.approx(0.4, abs=0.02)


def test_brownian_motion_variance_and_covariance():
    result = brownian_motion(n_paths=40000, n_steps=50, t_max=2.0, seed=0)
    assert result.times[0] == 0.0 and result.times[-1] == pytest.approx(2.0)
    assert np.all(result.paths[:, 0] == 0.0)
    w_half, w_end = result.paths[:, 25], result.paths[:, -1]
    assert w_end.var() == pytest.approx(2.0, rel=0.03)
    assert np.cov(w_half, w_end)[0, 1] == pytest.approx(1.0, rel=0.05)  # Cov(W_s, W_t) = min(s, t)


def test_brownian_quadratic_variation_equals_time():
    result = brownian_motion(n_paths=1, n_steps=200000, t_max=3.0, seed=1)
    assert np.sum(np.diff(result.paths[0]) ** 2) == pytest.approx(3.0, rel=0.02)


def test_simple_random_walk_steps_are_unit_and_centered():
    walks = simple_random_walk(n_walks=5000, n_steps=100, dim=3, seed=0)
    assert walks.shape == (5000, 101, 3)
    assert np.all(np.abs(np.diff(walks, axis=1)).sum(axis=2) == 1)
    assert np.mean(np.sum(walks[:, -1] ** 2, axis=1)) == pytest.approx(100.0, rel=0.05)  # E|S_n|^2 = n


def test_return_probability_1d_closed_form():
    for m in range(1, 8):
        assert return_probability_1d(2 * m) == pytest.approx(1 - math.comb(2 * m, m) / 4**m)
    assert return_probability_1d(10**6) > 0.999


def test_random_walk_return_fraction_matches_exact_1d_and_polya_3d():
    assert random_walk_return_fraction(1, 60, n_walks=40000, seed=2) == pytest.approx(return_probability_1d(60), abs=0.01)
    # Polya's 3D return probability is 0.3405; a finite horizon stays below it.
    frac3 = random_walk_return_fraction(3, 500, n_walks=20000, seed=3)
    assert 0.29 < frac3 < 0.3405 + 0.01


def test_erlang_b_matches_closed_form():
    for load in (0.5, 3.0, 12.0):
        for c in (0, 1, 4, 10):
            exact = (load**c / math.factorial(c)) / sum(load**i / math.factorial(i) for i in range(c + 1))
            assert erlang_b(load, c) == pytest.approx(exact, rel=1e-12)


def test_erlang_b_large_system_is_stable():
    b = erlang_b(900.0, 1000)
    assert 0.0 < b < 0.01
    assert erlang_b(0.0, 3) == 0.0


def test_ctmc_two_state_closed_form():
    a, b, t = 1.5, 0.5, 0.8
    P = ctmc_transition_matrix([[-a, a], [b, -b]], t)
    assert P[0, 1] == pytest.approx(a / (a + b) * (1 - np.exp(-(a + b) * t)))
    assert np.allclose(P.sum(axis=1), 1.0)


def test_ctmc_semigroup_and_stationarity():
    Q = np.array([[-3.0, 2.0, 1.0], [1.0, -1.5, 0.5], [0.5, 0.5, -1.0]])
    P1, P2 = ctmc_transition_matrix(Q, 0.4), ctmc_transition_matrix(Q, 0.9)
    assert np.allclose(P1 @ P2, ctmc_transition_matrix(Q, 1.3))  # Chapman-Kolmogorov
    pi = ctmc_stationary_distribution(Q)
    assert np.allclose(pi @ Q, 0.0) and pi.sum() == pytest.approx(1.0)
    assert np.allclose(ctmc_transition_matrix(Q, 50.0), np.tile(pi, (3, 1)))


def test_ctmc_rejects_invalid_generator():
    with pytest.raises(ValueError):
        ctmc_transition_matrix([[-1.0, 2.0], [1.0, -1.0]], 1.0)
