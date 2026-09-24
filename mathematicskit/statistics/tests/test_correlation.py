"""Tests for Pearson and Spearman correlation against closed forms."""

import numpy as np
import pytest
from scipy import stats

from mathematicskit.statistics.systems.correlation import pearson_correlation, spearman_correlation


def test_pearson_is_plus_minus_one_for_exact_lines():
    x = np.linspace(-3.0, 7.0, 25)
    assert pearson_correlation(x, 3.0 * x - 2.0).coefficient == pytest.approx(1.0)
    assert pearson_correlation(x, -0.5 * x + 4.0).coefficient == pytest.approx(-1.0)


def test_pearson_matches_definition_and_t_statistic_p_value():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = 0.4 * x + rng.normal(size=40)
    result = pearson_correlation(x, y)
    dx, dy = x - x.mean(), y - y.mean()
    r = np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    assert result.coefficient == pytest.approx(r)
    t = r * np.sqrt((40 - 2) / (1 - r**2))
    assert result.p_value == pytest.approx(2 * stats.t.sf(abs(t), 38))
    assert result.n == 40


def test_spearman_matches_rank_difference_formula_without_ties():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=30), rng.normal(size=30)
    d = stats.rankdata(x) - stats.rankdata(y)
    expected = 1 - 6 * np.sum(d**2) / (30 * (30**2 - 1))
    assert spearman_correlation(x, y).coefficient == pytest.approx(expected)


def test_spearman_is_one_for_monotone_but_pearson_is_not():
    x = np.linspace(0.0, 5.0, 20)
    assert spearman_correlation(x, np.exp(x)).coefficient == pytest.approx(1.0)
    assert pearson_correlation(x, np.exp(x)).coefficient < 0.95


def test_correlation_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pearson_correlation([1.0, 2.0, 3.0], [1.0, 2.0])
