"""mathematicskit.statistics: statistical inference.

Descriptive statistics (mean, variance, skewness, kurtosis, order
statistics) via ``numpy``/``scipy.stats``; hypothesis tests (one/two-
sample z- and t-tests, chi-square goodness-of-fit and independence
tests, one-way ANOVA) using ``mathematicskit.probability``'s ``Normal`` for
z-tests and ``scipy.stats``'s t/chi-square/F distributions elsewhere;
confidence intervals for means, proportions, and variances; ordinary
least-squares linear regression with residual diagnostics and
R^2/adjusted-R^2; and bootstrap resampling via
``scipy.stats.bootstrap`` for confidence intervals and standard errors.
"""

from mathematicskit.statistics.core.base import BootstrapResult, ConfidenceIntervalResult, DescriptiveStatsResult, HypothesisTestResult, RegressionResult
from mathematicskit.statistics.systems.bootstrap import bootstrap_confidence_interval
from mathematicskit.statistics.systems.confidence_intervals import mean_confidence_interval, proportion_confidence_interval, variance_confidence_interval
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
from mathematicskit.statistics.systems.regression import linear_regression
from mathematicskit.statistics.utils.effect_size import cohens_d

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DescriptiveStatsResult",
    "HypothesisTestResult",
    "ConfidenceIntervalResult",
    "RegressionResult",
    "BootstrapResult",
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
]
