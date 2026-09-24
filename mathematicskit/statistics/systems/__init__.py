"""Concrete descriptive statistics, hypothesis tests, confidence
intervals, regression, correlation, likelihood, resampling (bootstrap
and jackknife), shrinkage, and multiple-testing procedures."""

from mathematicskit.statistics.systems.bootstrap import bootstrap_confidence_interval
from mathematicskit.statistics.systems.confidence_intervals import mean_confidence_interval, proportion_confidence_interval, variance_confidence_interval
from mathematicskit.statistics.systems.correlation import pearson_correlation, spearman_correlation
from mathematicskit.statistics.systems.descriptive import descriptive_stats, order_statistic
from mathematicskit.statistics.systems.hypothesis_tests import (
    chi_square_goodness_of_fit,
    chi_square_independence,
    one_sample_t_test,
    one_sample_z_test,
    one_way_anova,
    two_sample_t_test,
    two_sample_z_test,
)
from mathematicskit.statistics.systems.jackknife import jackknife
from mathematicskit.statistics.systems.likelihood import likelihood_ratio_test, maximum_likelihood_fit
from mathematicskit.statistics.systems.multiple_testing import benjamini_hochberg, bonferroni_correction
from mathematicskit.statistics.systems.nonparametric import fisher_exact_test, kolmogorov_smirnov_test, mann_whitney_u_test, wilcoxon_signed_rank_test
from mathematicskit.statistics.systems.regression import linear_regression
from mathematicskit.statistics.systems.shrinkage import james_stein_estimator

__all__ = [
    "descriptive_stats",
    "order_statistic",
    "one_sample_z_test",
    "two_sample_z_test",
    "one_sample_t_test",
    "two_sample_t_test",
    "chi_square_goodness_of_fit",
    "chi_square_independence",
    "one_way_anova",
    "mean_confidence_interval",
    "proportion_confidence_interval",
    "variance_confidence_interval",
    "linear_regression",
    "bootstrap_confidence_interval",
    "pearson_correlation",
    "spearman_correlation",
    "jackknife",
    "maximum_likelihood_fit",
    "likelihood_ratio_test",
    "bonferroni_correction",
    "benjamini_hochberg",
    "kolmogorov_smirnov_test",
    "fisher_exact_test",
    "wilcoxon_signed_rank_test",
    "mann_whitney_u_test",
    "james_stein_estimator",
]
