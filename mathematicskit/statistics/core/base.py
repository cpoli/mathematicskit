"""Result containers for mathematicskit.statistics.

Every statistical procedure in this domain -- descriptive summaries,
hypothesis tests, confidence intervals, regression, and bootstrap
resampling -- returns one of the small dataclasses below (mirroring
physicskit's ``SimulationResult`` pattern), rather than a bare tuple or
dict, so visualizers and downstream code have a stable interface. This
domain has no natural "one ABC per algorithm family" the way, e.g.,
:mod:`mathematicskit.numerical_analysis` does (each test/interval/model in
``systems/`` is its own function with a distinct signature, not
interchangeable implementations of one interface), so unlike most other
domains' ``core/base.py`` there are no ABCs here -- following
physicskit's own precedent of varying internal shape by what a family of
algorithms actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "DescriptiveStatsResult",
    "HypothesisTestResult",
    "ConfidenceIntervalResult",
    "RegressionResult",
    "BootstrapResult",
]


@dataclass
class DescriptiveStatsResult:
    """Container for a dataset's descriptive-statistics summary."""

    n: int
    """int: Sample size."""

    mean: float
    """float: Sample mean."""

    variance: float
    """float: Sample variance (``ddof=1``, the unbiased estimator)."""

    std: float
    """float: Sample standard deviation (``ddof=1``)."""

    skewness: float
    """float: Sample skewness (Fisher-Pearson, ``scipy.stats.skew``)."""

    kurtosis: float
    """float: Sample excess kurtosis (``scipy.stats.kurtosis``, normal
    distribution has kurtosis 0 under this convention)."""

    minimum: float
    q1: float
    """float: First quartile (25th percentile)."""
    median: float
    q3: float
    """float: Third quartile (75th percentile)."""
    maximum: float


@dataclass
class HypothesisTestResult:
    """Container for the output of a hypothesis test."""

    statistic: float
    """float: The test statistic (e.g. ``z``, ``t``, :math:`\\chi^2`, ``F``)."""

    p_value: float
    """float: The p-value under the null hypothesis."""

    df: Optional[float] = None
    """float, optional: Degrees of freedom, where applicable."""

    method: str = ""
    """str: e.g. ``"one_sample_t"``, ``"chi_square_independence"``."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics slot (e.g. group means for ANOVA)."""

    def reject_null(self, alpha: float = 0.05) -> bool:
        """Whether the p-value falls below the significance level ``alpha``.

        Parameters
        ----------
        alpha : float

        Returns
        -------
        bool
        """
        return self.p_value < alpha


@dataclass
class ConfidenceIntervalResult:
    """Container for a confidence interval."""

    estimate: float
    """float: The point estimate (e.g. sample mean/proportion/variance)."""

    lower: float
    upper: float
    confidence_level: float = 0.95
    method: str = ""
    """str: e.g. ``"z"``, ``"t"``, ``"chi_square_variance"``, ``"wald_proportion"``."""

    def width(self) -> float:
        """float: ``upper - lower``."""
        return self.upper - self.lower


@dataclass
class RegressionResult:
    """Container for an ordinary-least-squares regression fit."""

    coefficients: np.ndarray
    """ndarray, shape (p,): Fitted coefficients (intercept first, if included)."""

    standard_errors: np.ndarray
    """ndarray, shape (p,): Standard error of each coefficient."""

    t_statistics: np.ndarray
    """ndarray, shape (p,): Per-coefficient t-statistic for ``H0: beta_j = 0``."""

    p_values: np.ndarray
    """ndarray, shape (p,): Per-coefficient two-sided p-value."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adjusted_r_squared: float


@dataclass
class BootstrapResult:
    """Container for a bootstrap confidence interval/standard error."""

    estimate: float
    """float: The statistic computed on the original (non-resampled) data."""

    lower: float
    upper: float
    std_error: float
    confidence_level: float = 0.95
    method: str = "BCa"
    """str: The bootstrap CI method (``scipy.stats.bootstrap``'s
    ``"percentile"``, ``"basic"``, or ``"BCa"``)."""
