"""Tests for Kolmogorov-Smirnov, Fisher's exact, Wilcoxon, and Mann-Whitney tests."""

from math import comb

import numpy as np
import pytest
from scipy import stats

from mathematicskit.statistics.systems.nonparametric import (
    fisher_exact_test,
    kolmogorov_smirnov_test,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
)


def test_ks_statistic_matches_largest_ecdf_gap():
    rng = np.random.default_rng(0)
    data = np.sort(rng.normal(size=50))
    cdf = stats.norm.cdf(data)
    i = np.arange(1, 51)
    expected = max(np.max(i / 50 - cdf), np.max(cdf - (i - 1) / 50))
    assert kolmogorov_smirnov_test(data, "norm").statistic == pytest.approx(expected)


def test_ks_accepts_correct_model_and_rejects_shifted_one():
    rng = np.random.default_rng(4)
    data = rng.normal(size=400)
    assert kolmogorov_smirnov_test(data, "norm").p_value > 0.2
    assert kolmogorov_smirnov_test(data, "norm", args=(0.5, 1.0)).reject_null()


def test_ks_two_sample():
    rng = np.random.default_rng(2)
    result = kolmogorov_smirnov_test(rng.normal(size=300), rng.normal(loc=1.0, size=300))
    assert result.method == "kolmogorov_smirnov_2samp"
    assert result.reject_null(alpha=0.001)


def test_lady_tasting_tea_p_value_is_one_over_seventy():
    result = fisher_exact_test([[4, 0], [0, 4]], alternative="greater")
    assert result.p_value == pytest.approx(1 / comb(8, 4))
    # three of four right: (C(4,3)C(4,1) + 1) / 70
    assert fisher_exact_test([[3, 1], [1, 3]], alternative="greater").p_value == pytest.approx(17 / 70)


def test_fisher_exact_odds_ratio_and_shape_check():
    assert fisher_exact_test([[6, 2], [3, 4]]).statistic == pytest.approx(6 * 4 / (2 * 3))
    with pytest.raises(ValueError):
        fisher_exact_test([[1, 2, 3], [4, 5, 6]])


def test_wilcoxon_exact_all_positive_differences():
    assert wilcoxon_signed_rank_test(np.arange(1.0, 6.0), alternative="greater").p_value == pytest.approx(1 / 32)
    assert wilcoxon_signed_rank_test(np.arange(1.0, 6.0)).p_value == pytest.approx(2 / 32)


def test_wilcoxon_paired_equals_differences():
    x = np.array([5.1, 4.8, 6.0, 5.5, 5.9, 6.2, 4.9])
    y = np.array([4.9, 4.9, 5.2, 5.0, 5.1, 5.6, 4.1])
    assert wilcoxon_signed_rank_test(x, y).p_value == pytest.approx(wilcoxon_signed_rank_test(x - y).p_value)


def test_mann_whitney_complete_separation():
    x, y = np.array([5.0, 6.0, 7.0, 8.0]), np.array([1.0, 2.0, 3.0, 4.0])
    result = mann_whitney_u_test(x, y)
    assert result.statistic == 16.0
    assert result.p_value == pytest.approx(2 / comb(8, 4))


def test_mann_whitney_u_counts_winning_pairs():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=12), rng.normal(size=9)
    assert mann_whitney_u_test(x, y).statistic == float(np.sum(x[:, None] > y[None, :]))
