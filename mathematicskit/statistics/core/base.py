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
    "CorrelationResult",
    "MaximumLikelihoodResult",
    "JackknifeResult",
    "MultipleTestingResult",
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
    """float: The smallest observation."""

    q1: float
    """float: First quartile (25th percentile)."""

    median: float
    """float: The 50th percentile."""

    q3: float
    """float: Third quartile (75th percentile)."""

    maximum: float
    """float: The largest observation. Together with `minimum`, `q1`,
    `median` and `q3` this makes up the five-number summary."""


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
    """float: The interval's lower endpoint."""

    upper: float
    """float: The interval's upper endpoint."""

    confidence_level: float = 0.95
    """float: The interval's nominal coverage probability. The endpoints
    come from the :math:`\\alpha/2` and :math:`1-\\alpha/2` quantiles of the
    reference distribution, with :math:`\\alpha = 1 - \\text{confidence
    level}`."""

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
    """ndarray, shape (n,): :math:`X\\hat\\beta`, the model's prediction at
    each observed predictor."""

    residuals: np.ndarray
    """ndarray, shape (n,): ``y - fitted_values``."""

    r_squared: float
    """float: The fraction of the response's variance the fit explains,
    :math:`1 - SS_{\\text{res}}/SS_{\\text{tot}}`; ``1.0`` for a perfect
    fit."""

    adjusted_r_squared: float
    """float: `r_squared` penalized for the number of predictors,
    :math:`1 - (1-R^2)(n-1)/(n-p)`. Unlike `r_squared`, it does not rise
    automatically each time another predictor is added, so it is the
    fairer figure when comparing models of different size."""


@dataclass
class BootstrapResult:
    """Container for a bootstrap confidence interval/standard error."""

    estimate: float
    """float: The statistic computed on the original (non-resampled) data."""

    lower: float
    """float: Lower endpoint of the bootstrap confidence interval."""

    upper: float
    """float: Upper endpoint of the bootstrap confidence interval."""

    std_error: float
    """float: The bootstrap distribution's standard deviation -- the
    resampling estimate of `estimate`'s standard error, obtained without
    any closed-form sampling-distribution assumption."""

    confidence_level: float = 0.95
    """float: The interval's nominal coverage probability."""

    method: str = "BCa"
    """str: The bootstrap CI method (``scipy.stats.bootstrap``'s
    ``"percentile"``, ``"basic"``, or ``"BCa"``)."""


@dataclass
class CorrelationResult:
    """Container for a correlation coefficient and its significance test."""

    coefficient: float
    """float: The correlation coefficient, in :math:`[-1, 1]`."""

    p_value: float
    """float: Two-sided p-value for :math:`H_0`: no association."""

    n: int
    """int: Number of paired observations."""

    method: str = ""
    """str: ``"pearson"`` or ``"spearman"``."""


@dataclass
class MaximumLikelihoodResult:
    """Container for a maximum-likelihood distribution fit."""

    params: tuple
    """tuple: The fitted parameters, in ``scipy.stats`` order (shape
    parameters first, then ``loc`` and ``scale``)."""

    log_likelihood: float
    """float: The maximized log-likelihood :math:`\\ell(\\hat\\theta)`."""

    n_params: int
    """int: Number of parameters actually estimated (fixed ones excluded)."""

    distribution: str = ""
    """str: The ``scipy.stats`` distribution name, e.g. ``"norm"``."""

    def aic(self) -> float:
        """float: Akaike's information criterion, :math:`2k - 2\\ell(\\hat\\theta)`."""
        return 2.0 * self.n_params - 2.0 * self.log_likelihood


@dataclass
class JackknifeResult:
    """Container for a jackknife bias and standard-error estimate."""

    estimate: float
    """float: The statistic computed on the full sample."""

    bias: float
    """float: Quenouille's jackknife bias estimate."""

    std_error: float
    """float: Tukey's jackknife standard-error estimate."""

    bias_corrected: float
    """float: ``estimate - bias``."""

    replicates: np.ndarray
    """ndarray, shape (n,): The leave-one-out values of the statistic."""


@dataclass
class MultipleTestingResult:
    """Container for a multiple-testing correction."""

    rejected: np.ndarray
    """ndarray of bool, shape (m,): Which null hypotheses are rejected."""

    adjusted_p_values: np.ndarray
    """ndarray, shape (m,): Adjusted p-values, comparable directly with `alpha`."""

    alpha: float = 0.05
    """float: The error rate being controlled (family-wise error rate or
    false discovery rate, depending on `method`)."""

    method: str = ""
    """str: ``"bonferroni"`` or ``"benjamini_hochberg"``."""
