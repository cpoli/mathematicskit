"""Tests for Monte Carlo integration (plain, importance sampling, and
control variates) against closed-form integrals."""

import numpy as np
import pytest

from mathematicskit.probability.systems.continuous import Exponential
from mathematicskit.probability.systems.monte_carlo import control_variates_integrate, importance_sampling_integrate, monte_carlo_integrate
from mathematicskit.probability.utils.diagnostics import effective_sample_size


def test_plain_monte_carlo_matches_closed_form_integral():
    result = monte_carlo_integrate(lambda x: x**2, 0.0, 1.0, n=300000, seed=0)
    assert result.estimate == pytest.approx(1.0 / 3.0, abs=0.01)
    assert result.n_samples == 300000
    assert result.method == "plain"


def test_importance_sampling_matches_closed_form_integral():
    proposal = Exponential(rate=1.0)
    f = lambda x: np.exp(-(x**2) / 2.0)
    result = importance_sampling_integrate(f, lambda rng, n: proposal.sample(size=n, seed=rng), proposal.pdf, n=300000, seed=1)
    assert result.estimate == pytest.approx(np.sqrt(np.pi / 2.0), abs=0.02)


def test_control_variates_matches_closed_form_integral():
    result = control_variates_integrate(np.exp, lambda x: x, control_mean=0.5, a=0.0, b=1.0, n=300000, seed=2)
    assert result.estimate == pytest.approx(np.e - 1.0, abs=0.01)


def test_control_variates_reduces_standard_error_vs_plain():
    plain = monte_carlo_integrate(np.exp, 0.0, 1.0, n=50000, seed=3)
    cv = control_variates_integrate(np.exp, lambda x: x, control_mean=0.5, a=0.0, b=1.0, n=50000, seed=3)
    assert cv.std_error < plain.std_error


def test_effective_sample_size_bounds():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    assert effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_effective_sample_size_is_lower_for_skewed_weights():
    uniform_weights = np.ones(100)
    skewed_weights = np.concatenate([[100.0], np.ones(99) * 0.01])
    assert effective_sample_size(skewed_weights) < effective_sample_size(uniform_weights)
