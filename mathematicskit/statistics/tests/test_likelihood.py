"""Tests for maximum-likelihood fitting and the likelihood-ratio test."""

import numpy as np
import pytest
from scipy import stats

from mathematicskit.statistics.systems.likelihood import likelihood_ratio_test, maximum_likelihood_fit


def test_normal_mle_is_mean_and_biased_standard_deviation():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=500)
    result = maximum_likelihood_fit(data, "norm")
    assert result.params[0] == pytest.approx(np.mean(data))
    assert result.params[1] == pytest.approx(np.std(data, ddof=0))
    assert result.n_params == 2
    assert result.log_likelihood == pytest.approx(np.sum(stats.norm.logpdf(data, *result.params)))


def test_normal_log_likelihood_closed_form():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = maximum_likelihood_fit(data, stats.norm)
    sigma2 = np.var(data)
    assert result.log_likelihood == pytest.approx(-0.5 * 5 * (np.log(2 * np.pi * sigma2) + 1))
    assert result.aic() == pytest.approx(4 - 2 * result.log_likelihood)


def test_exponential_mle_scale_is_sample_mean_with_fixed_location():
    rng = np.random.default_rng(1)
    data = rng.exponential(scale=4.0, size=400)
    result = maximum_likelihood_fit(data, "expon", floc=0)
    assert result.params[0] == 0.0
    assert result.params[1] == pytest.approx(np.mean(data), rel=1e-6)
    assert result.n_params == 1


def test_likelihood_ratio_critical_value_gives_alpha():
    critical = stats.chi2.ppf(0.95, 2)
    result = likelihood_ratio_test(0.0, critical / 2, df=2)
    assert result.statistic == pytest.approx(critical)
    assert result.p_value == pytest.approx(0.05)


def test_wilks_nested_normal_models():
    rng = np.random.default_rng(2)
    data = rng.normal(loc=0.5, scale=1.0, size=200)
    null = maximum_likelihood_fit(data, "norm", floc=0.0)
    alt = maximum_likelihood_fit(data, "norm")
    result = likelihood_ratio_test(null.log_likelihood, alt.log_likelihood, df=alt.n_params - null.n_params)
    # closed form for this pair of models: n log(sigma0^2 / sigma1^2)
    expected = 200 * np.log(np.mean(data**2) / np.var(data))
    assert result.statistic == pytest.approx(expected, rel=1e-6)
    assert result.reject_null(alpha=0.01)


def test_likelihood_ratio_rejects_bad_df():
    with pytest.raises(ValueError):
        likelihood_ratio_test(-1.0, 0.0, df=0)
