"""Tests for the Law of Large Numbers and Central Limit Theorem
simulations against their theoretical predictions."""

from scipy import stats

from mathematicskit.probability.systems.continuous import Exponential, Uniform
from mathematicskit.probability.systems.discrete import Poisson
from mathematicskit.probability.systems.limit_theorems import central_limit_theorem_sample_means, law_of_large_numbers_trace


def test_lln_sample_mean_converges_to_true_mean():
    dist = Exponential(rate=2.0)
    means = law_of_large_numbers_trace(dist, n_values=[100, 100000], seed=0)
    assert abs(means[-1] - dist.mean) < abs(means[0] - dist.mean)
    assert abs(means[-1] - dist.mean) < 0.02


def test_lln_matches_requested_sample_sizes_count():
    dist = Uniform(a=0.0, b=1.0)
    means = law_of_large_numbers_trace(dist, n_values=[10, 100, 1000, 10000], seed=1)
    assert means.shape == (4,)


def test_clt_standardized_means_have_zero_mean_unit_variance():
    dist = Poisson(mu=5.0)
    z = central_limit_theorem_sample_means(dist, n=300, n_trials=5000, seed=2)
    assert abs(float(z.mean())) < 0.1
    assert abs(float(z.std()) - 1.0) < 0.1


def test_clt_standardized_means_pass_normality_goodness_of_fit():
    """Cross-check against scipy.stats: a Kolmogorov-Smirnov test should not
    strongly reject standard normality for a reasonably large n."""
    dist = Uniform(a=0.0, b=1.0)
    z = central_limit_theorem_sample_means(dist, n=200, n_trials=3000, seed=3)
    _, p_value = stats.kstest(z, "norm")
    assert p_value > 0.01
