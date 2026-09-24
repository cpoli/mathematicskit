"""Tests for Buffon's needle, the St. Petersburg game, Bayes's
Beta-binomial update, and Chebyshev's inequality against closed forms."""

import numpy as np
import pytest
from scipy import integrate, stats

from mathematicskit.probability.systems.bayes import beta_binomial_posterior, rule_of_succession
from mathematicskit.probability.systems.buffon import buffon_needle
from mathematicskit.probability.systems.continuous import Beta, Exponential, Normal, Uniform
from mathematicskit.probability.systems.discrete import Poisson
from mathematicskit.probability.systems.inequalities import chebyshev_tail
from mathematicskit.probability.systems.st_petersburg import st_petersburg_certainty_equivalent, st_petersburg_payoffs


@pytest.mark.parametrize("length, spacing", [(1.0, 1.0), (0.5, 1.0), (1.0, 3.0)])
def test_buffon_crossing_fraction_matches_2l_over_pi_d(length, spacing):
    result = buffon_needle(n_drops=400000, length=length, spacing=spacing, seed=1)
    assert result.exact_crossing_probability == pytest.approx(2 * length / (np.pi * spacing))
    se = np.sqrt(result.exact_crossing_probability * (1 - result.exact_crossing_probability) / result.n_drops)
    assert abs(result.crossing_fraction - result.exact_crossing_probability) < 4 * se


def test_buffon_pi_estimate():
    assert buffon_needle(n_drops=1000000, seed=2).pi_estimate == pytest.approx(np.pi, abs=0.02)


def test_buffon_rejects_long_needle():
    with pytest.raises(ValueError):
        buffon_needle(length=2.0, spacing=1.0)


def test_st_petersburg_payoff_distribution():
    payoffs = st_petersburg_payoffs(200000, seed=0)
    assert np.all(np.log2(payoffs) == np.round(np.log2(payoffs)))
    assert np.mean(payoffs == 2.0) == pytest.approx(0.5, abs=0.005)
    assert np.mean(payoffs == 4.0) == pytest.approx(0.25, abs=0.005)


def test_st_petersburg_certainty_equivalent_closed_form():
    # w = 0: ln c = sum_k k ln2 / 2^k = 2 ln 2, so c = 4.
    assert st_petersburg_certainty_equivalent(0.0) == pytest.approx(4.0, rel=1e-12)
    # Richer players value the game more, but it stays finite.
    values = [st_petersburg_certainty_equivalent(w) for w in (0.0, 10.0, 1000.0)]
    assert values[0] < values[1] < values[2] < 30.0


def test_beta_mean_variance_and_mgf():
    b = Beta(alpha=2.0, beta=5.0)
    assert b.mean == pytest.approx(2 / 7)
    assert b.variance == pytest.approx(2 * 5 / (7**2 * 8))
    expected, _ = integrate.quad(lambda x: np.exp(0.7 * x) * b.pdf(x), 0.0, 1.0)
    assert float(b.mgf(0.7)) == pytest.approx(expected, rel=1e-8)


def test_beta_binomial_posterior_matches_normalized_likelihood():
    k, n = 3, 12
    post = beta_binomial_posterior(k, n, prior_alpha=2.0, prior_beta=2.0)
    assert (post.alpha, post.beta) == (5.0, 11.0)
    unnormalized = lambda p: p**k * (1 - p) ** (n - k) * stats.beta(2, 2).pdf(p)
    z, _ = integrate.quad(unnormalized, 0.0, 1.0)
    for p in (0.1, 0.3, 0.6):
        assert post.pdf(p) == pytest.approx(unnormalized(p) / z, rel=1e-8)


def test_rule_of_succession():
    assert rule_of_succession(0, 0) == pytest.approx(0.5)
    assert rule_of_succession(10, 10) == pytest.approx(11 / 12)
    with pytest.raises(ValueError):
        beta_binomial_posterior(5, 3)


@pytest.mark.parametrize("dist", [Normal(0.0, 1.0), Exponential(rate=2.0), Uniform(0.0, 1.0), Poisson(mu=4.0)])
def test_chebyshev_bound_holds(dist):
    result = chebyshev_tail(dist, np.linspace(0.5, 5.0, 20))
    assert np.all(result.exact <= result.bound + 1e-12)


def test_chebyshev_exact_tail_closed_forms():
    k = np.array([1.0, 2.0, 3.0])
    assert chebyshev_tail(Normal(0.0, 1.0), k).exact == pytest.approx(2 * stats.norm.sf(k))
    # Exponential(1): mean = std = 1, so P(|X-1| >= k) = e^{-(1+k)} for k >= 1.
    assert chebyshev_tail(Exponential(rate=1.0), k).exact == pytest.approx(np.exp(-(1 + k)))


def test_chebyshev_discrete_tail_counts_boundary_points():
    # Poisson(4): mean 4, std 2; k = 1 gives P(X <= 2) + P(X >= 6).
    exact = chebyshev_tail(Poisson(mu=4.0), 1.0).exact[0]
    pois = stats.poisson(4.0)
    assert exact == pytest.approx(pois.cdf(2) + pois.sf(5))
