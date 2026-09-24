"""mathematicskit.statistics: statistical inference.

Descriptive statistics (mean, variance, skewness, kurtosis, order
statistics) via ``numpy``/``scipy.stats``; hypothesis tests (one/two-
sample z- and t-tests, chi-square goodness-of-fit and independence
tests, one-way ANOVA) using ``mathematicskit.probability``'s ``Normal`` for
z-tests and ``scipy.stats``'s t/chi-square/F distributions elsewhere;
confidence intervals for means, proportions, and variances; ordinary
least-squares linear regression with residual diagnostics and
R^2/adjusted-R^2; bootstrap resampling via ``scipy.stats.bootstrap``
and the leave-one-out jackknife for confidence intervals, bias, and
standard errors; Pearson and Spearman correlation; maximum-likelihood
fitting and the likelihood-ratio test; distribution-free and exact tests
(Kolmogorov-Smirnov, Fisher's exact, Wilcoxon signed-rank,
Mann-Whitney U); James-Stein shrinkage; and Bonferroni and
Benjamini-Hochberg multiple-testing corrections.
"""

from mathematicskit.statistics.core.base import (
    BootstrapResult,
    ConfidenceIntervalResult,
    CorrelationResult,
    DescriptiveStatsResult,
    HypothesisTestResult,
    JackknifeResult,
    MaximumLikelihoodResult,
    MultipleTestingResult,
    RegressionResult,
)
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
from mathematicskit.statistics.utils.effect_size import cohens_d

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DescriptiveStatsResult",
    "HypothesisTestResult",
    "ConfidenceIntervalResult",
    "RegressionResult",
    "BootstrapResult",
    "CorrelationResult",
    "MaximumLikelihoodResult",
    "JackknifeResult",
    "MultipleTestingResult",
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
    "cohens_d",
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
