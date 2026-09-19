"""Tests for Markov chains against closed-form stationary distributions
and gambler's-ruin absorption probabilities."""

import numpy as np
import pytest

from mathkit.probability.systems.markov_chain import MarkovChain


def test_stationary_distribution_matches_closed_form_two_state_chain():
    p = np.array([[0.9, 0.1], [0.5, 0.5]])
    chain = MarkovChain(p)
    pi = chain.stationary_distribution()
    np.testing.assert_allclose(pi, [5.0 / 6.0, 1.0 / 6.0], atol=1e-8)
    assert pi.sum() == pytest.approx(1.0)


def test_power_iteration_agrees_with_eigen_method():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    chain = MarkovChain(p)
    pi_eig = chain.stationary_distribution()
    pi_power = chain.stationary_distribution_power_iteration()
    np.testing.assert_allclose(pi_eig, pi_power, atol=1e-6)


def test_stationary_distribution_is_a_fixed_point():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    chain = MarkovChain(p)
    pi = chain.stationary_distribution()
    np.testing.assert_allclose(pi @ p, pi, atol=1e-8)


def _gamblers_ruin(n_capital=4):
    size = n_capital + 1
    p = np.zeros((size, size))
    p[0, 0] = 1.0
    p[-1, -1] = 1.0
    for i in range(1, n_capital):
        p[i, i - 1] = 0.5
        p[i, i + 1] = 0.5
    return p


def test_gamblers_ruin_absorption_probabilities_match_closed_form():
    chain = MarkovChain(_gamblers_ruin(4))
    transient = [1, 2, 3]
    b = chain.absorption_probabilities(transient, absorbing=[0, 4])
    # P(reach N | start at i) = i / N for a fair random walk.
    expected_reach_4 = np.array([1, 2, 3]) / 4.0
    np.testing.assert_allclose(b[:, 1], expected_reach_4, atol=1e-8)
    np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-8)


def test_gamblers_ruin_expected_steps_match_closed_form():
    chain = MarkovChain(_gamblers_ruin(4))
    t = chain.expected_steps_to_absorption(transient=[1, 2, 3])
    expected = np.array([i * (4 - i) for i in (1, 2, 3)], dtype=float)
    np.testing.assert_allclose(t, expected, atol=1e-8)


def test_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        MarkovChain(np.ones((2, 3)))


def test_rejects_rows_not_summing_to_one():
    with pytest.raises(ValueError):
        MarkovChain(np.array([[0.5, 0.4], [0.5, 0.5]]))
