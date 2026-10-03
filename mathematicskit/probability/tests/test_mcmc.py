"""Tests for the Metropolis-Hastings algorithm and the Gibbs sampler."""

import numpy as np
import pytest
from scipy import stats

from mathematicskit.probability.systems.mcmc import gibbs_sampler, metropolis_hastings


def test_random_walk_metropolis_recovers_normal_moments():
    log_target = lambda x: -0.5 * float(((x - 2.0) / 3.0) @ ((x - 2.0) / 3.0))  # noqa: E731
    result = metropolis_hastings(log_target, [0.0], n_samples=60000, proposal_scale=7.0, burn_in=1000, seed=1)
    assert result.samples.shape == (60000, 1)
    assert result.method == "metropolis_hastings"
    assert result.samples.mean() == pytest.approx(2.0, abs=0.1)
    assert result.samples.std() == pytest.approx(3.0, rel=0.03)
    assert 0.2 < result.acceptance_rate < 0.6


def test_unnormalized_bimodal_target_visits_both_modes():
    log_target = lambda x: float(np.logaddexp(-0.5 * (x[0] + 3) ** 2, -0.5 * (x[0] - 3) ** 2))  # noqa: E731
    result = metropolis_hastings(log_target, [0.0], n_samples=60000, proposal_scale=4.0, seed=2)
    assert np.mean(result.samples[:, 0] > 0) == pytest.approx(0.5, abs=0.05)


def test_hastings_correction_for_asymmetric_proposal():
    # Target Gamma(3, 1); multiplicative log-normal proposal y = x e^{s Z} has q(y|x) ∝ 1/y.
    log_target = lambda x: float(2 * np.log(x[0]) - x[0]) if x[0] > 0 else -np.inf  # noqa: E731
    proposal = lambda x, rng: x * np.exp(0.8 * rng.standard_normal())  # noqa: E731
    log_q = lambda y, x: float(-np.log(y[0]))  # noqa: E731
    corrected = metropolis_hastings(log_target, [1.0], n_samples=60000, proposal=proposal, log_proposal_density=log_q, seed=3)
    assert corrected.samples.mean() == pytest.approx(3.0, rel=0.04)
    # Omitting the correction samples x e^{-x} instead (an extra factor of 1/x): Gamma(2, 1), mean 2.
    naive = metropolis_hastings(log_target, [1.0], n_samples=60000, proposal=proposal, seed=3)
    assert naive.samples.mean() == pytest.approx(2.0, rel=0.05)


def test_metropolis_rejects_infinite_start():
    with pytest.raises(ValueError):
        metropolis_hastings(lambda x: -np.inf, [0.0])


def test_gibbs_bivariate_normal_correlation_and_marginals():
    rho = -0.6
    s = np.sqrt(1 - rho**2)
    conditionals = [lambda x, rng: rho * x[1] + s * rng.standard_normal(), lambda x, rng: rho * x[0] + s * rng.standard_normal()]
    result = gibbs_sampler(conditionals, [5.0, -5.0], n_samples=50000, burn_in=100, seed=0)
    assert result.method == "gibbs" and result.acceptance_rate == 1.0
    assert np.corrcoef(result.samples.T)[0, 1] == pytest.approx(rho, abs=0.02)
    assert stats.kstest(result.samples[::10, 0], "norm").pvalue > 0.01


def test_gibbs_beta_binomial_marginal():
    # x | p ~ Binomial(n, p), p | x ~ Beta(x + a, n - x + b): the x-marginal is Beta-binomial (Casella and George, 1992).
    n, a, b = 16, 2.0, 4.0
    conditionals = [lambda z, rng: rng.binomial(n, z[1]), lambda z, rng: rng.beta(z[0] + a, n - z[0] + b)]
    result = gibbs_sampler(conditionals, [0.0, 0.5], n_samples=40000, seed=4)
    assert result.samples[:, 0].mean() == pytest.approx(stats.betabinom(n, a, b).mean(), rel=0.03)


def test_gibbs_requires_one_sampler_per_coordinate():
    with pytest.raises(ValueError):
        gibbs_sampler([lambda x, rng: 0.0], [0.0, 0.0])


def test_autocorrelation_time_of_ar1_and_iid():
    from scipy.signal import lfilter

    from mathematicskit.probability.utils.diagnostics import chain_effective_sample_size, integrated_autocorrelation_time

    rng = np.random.default_rng(5)
    for phi in (0.5, 0.9):
        x = lfilter([1.0], [1.0, -phi], rng.standard_normal(300000))
        assert integrated_autocorrelation_time(x) == pytest.approx((1 + phi) / (1 - phi), rel=0.08)
    iid = rng.standard_normal(50000)
    assert chain_effective_sample_size(iid) == pytest.approx(50000, rel=0.1)


def test_small_steps_mix_slowly():
    from mathematicskit.probability.utils.diagnostics import chain_effective_sample_size

    log_target = lambda x: -0.5 * float(x @ x)  # noqa: E731
    slow = metropolis_hastings(log_target, [0.0], n_samples=20000, proposal_scale=0.1, seed=6)
    good = metropolis_hastings(log_target, [0.0], n_samples=20000, proposal_scale=2.4, seed=6)
    assert slow.acceptance_rate > good.acceptance_rate
    assert chain_effective_sample_size(good.samples[:, 0]) > 10 * chain_effective_sample_size(slow.samples[:, 0])
