"""Tests for hypothesis tests against known/closed-form statistical results."""

import numpy as np
import pytest
from scipy import stats

from mathkit.statistics.systems.hypothesis_tests import (
    chi_square_goodness_of_fit,
    chi_square_independence,
    one_sample_t_test,
    one_sample_z_test,
    one_way_anova,
    two_sample_t_test,
    two_sample_z_test,
)


def test_one_sample_z_test_matches_hand_computed_statistic():
    data = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    result = one_sample_z_test(data, mu0=5.0, sigma=2.0)
    expected_z = (7.0 - 5.0) / (2.0 / np.sqrt(5))
    assert result.statistic == pytest.approx(expected_z)


def test_two_sample_z_test_zero_when_means_equal():
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([2.0, 2.0, 2.0])
    result = two_sample_z_test(data1, data2, sigma1=1.0, sigma2=1.0)
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_one_sample_t_test_matches_scipy():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=2.0, scale=1.0, size=30)
    result = one_sample_t_test(data, mu0=0.0)
    scipy_result = stats.ttest_1samp(data, 0.0)
    assert result.statistic == pytest.approx(scipy_result.statistic)
    assert result.p_value == pytest.approx(scipy_result.pvalue, abs=1e-8)


def test_two_sample_t_test_equal_var_matches_scipy():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, size=40)
    b = rng.normal(loc=0.5, size=45)
    result = two_sample_t_test(a, b, equal_var=True)
    scipy_result = stats.ttest_ind(a, b, equal_var=True)
    assert result.statistic == pytest.approx(scipy_result.statistic)
    assert result.p_value == pytest.approx(scipy_result.pvalue, abs=1e-8)


def test_welch_t_test_matches_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(loc=0.0, scale=1.0, size=30)
    b = rng.normal(loc=0.3, scale=3.0, size=50)
    result = two_sample_t_test(a, b, equal_var=False)
    scipy_result = stats.ttest_ind(a, b, equal_var=False)
    assert result.statistic == pytest.approx(scipy_result.statistic)
    assert result.df == pytest.approx(scipy_result.df, rel=1e-6)


def test_chi_square_goodness_of_fit_matches_scipy():
    observed = np.array([90.0, 105.0, 95.0, 100.0, 110.0, 100.0])
    expected = np.full(6, 100.0)
    result = chi_square_goodness_of_fit(observed, expected)
    scipy_result = stats.chisquare(observed, expected)
    assert result.statistic == pytest.approx(scipy_result.statistic)
    assert result.p_value == pytest.approx(scipy_result.pvalue, abs=1e-8)


def test_chi_square_independence_matches_scipy():
    table = np.array([[30.0, 10.0], [20.0, 40.0]])
    result = chi_square_independence(table)
    scipy_stat, scipy_p, scipy_df, _ = stats.chi2_contingency(table, correction=False)
    assert result.statistic == pytest.approx(scipy_stat)
    assert result.p_value == pytest.approx(scipy_p, abs=1e-8)
    assert result.df == scipy_df


def test_one_way_anova_matches_scipy():
    a = np.array([4.0, 5.0, 6.0, 5.0])
    b = np.array([7.0, 8.0, 7.0, 9.0])
    c = np.array([4.0, 3.0, 5.0, 4.0])
    result = one_way_anova(a, b, c)
    scipy_result = stats.f_oneway(a, b, c)
    assert result.statistic == pytest.approx(scipy_result.statistic)
    assert result.p_value == pytest.approx(scipy_result.pvalue, abs=1e-8)


def test_one_way_anova_requires_at_least_two_groups():
    with pytest.raises(ValueError):
        one_way_anova(np.array([1.0, 2.0]))


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_alternative_hypotheses_are_valid_probabilities(alternative):
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = one_sample_z_test(data, mu0=3.0, sigma=1.0, alternative=alternative)
    assert 0.0 <= result.p_value <= 1.0
