r"""Confidence intervals for means, proportions, and variances.

Reference distributions: :class:`mathematicskit.probability.Normal` for the
(known-variance) mean and proportion intervals, :class:`scipy.stats.t`
for the (unknown-variance) mean interval, and :class:`scipy.stats.chi2`
for the variance interval -- the quantile-function calls themselves are
the whole interval construction, no reason to hand-roll them. See
DeGroot & Schervish, *Probability and Statistics*, 4th ed., Ch. 9.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from mathematicskit.probability import Normal
from mathematicskit.statistics.core.base import ConfidenceIntervalResult

__all__ = ["mean_confidence_interval", "proportion_confidence_interval", "variance_confidence_interval"]

_STANDARD_NORMAL = Normal(mu=0.0, sigma=1.0)


def mean_confidence_interval(data: np.ndarray, sigma: Optional[float] = None, confidence_level: float = 0.95) -> ConfidenceIntervalResult:
    r"""Confidence interval for a population mean.

    :math:`\bar x \pm z_{\alpha/2}\,\sigma/\sqrt n` if `sigma` (the
    population standard deviation) is known, else :math:`\bar x \pm
    t_{\alpha/2,\,n-1}\,s/\sqrt n` using the sample standard deviation.
    See DeGroot & Schervish, *Probability and Statistics*, 4th ed.,
    Sec. 8.5-8.6.

    Parameters
    ----------
    data : array-like, shape (n,)
    sigma : float, optional
        Known population standard deviation; if omitted, the sample
        standard deviation and Student's t distribution are used instead.
    confidence_level : float

    Returns
    -------
    ConfidenceIntervalResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(loc=10.0, scale=2.0, size=100)
    >>> result = mean_confidence_interval(data, sigma=2.0)
    >>> bool(result.lower < 10.0 < result.upper)
    True
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    mean = float(np.mean(x))
    alpha = 1.0 - confidence_level
    if sigma is not None:
        z = _STANDARD_NORMAL.ppf(1.0 - alpha / 2.0)
        margin = z * sigma / np.sqrt(n)
        method = "z"
    else:
        df = n - 1
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        margin = t_crit * float(np.std(x, ddof=1)) / np.sqrt(n)
        method = "t"
    return ConfidenceIntervalResult(estimate=mean, lower=mean - margin, upper=mean + margin, confidence_level=confidence_level, method=method)


def proportion_confidence_interval(successes: int, n: int, confidence_level: float = 0.95) -> ConfidenceIntervalResult:
    r"""Wald (normal-approximation) confidence interval for a population proportion.

    :math:`\hat p \pm z_{\alpha/2}\sqrt{\hat p(1-\hat p)/n}`, valid for
    ``n`` large enough that the normal approximation to the binomial
    holds (a common rule of thumb: :math:`n\hat p \geq 5` and
    :math:`n(1-\hat p) \geq 5`). See DeGroot & Schervish, *Probability
    and Statistics*, 4th ed., Sec. 9.5.

    Parameters
    ----------
    successes : int
    n : int
    confidence_level : float

    Returns
    -------
    ConfidenceIntervalResult

    Examples
    --------
    >>> result = proportion_confidence_interval(successes=520, n=1000)
    >>> round(result.estimate, 4)
    0.52
    """
    p_hat = successes / n
    alpha = 1.0 - confidence_level
    z = _STANDARD_NORMAL.ppf(1.0 - alpha / 2.0)
    margin = z * np.sqrt(p_hat * (1.0 - p_hat) / n)
    return ConfidenceIntervalResult(
        estimate=p_hat, lower=max(0.0, p_hat - margin), upper=min(1.0, p_hat + margin), confidence_level=confidence_level, method="wald_proportion"
    )


def variance_confidence_interval(data: np.ndarray, confidence_level: float = 0.95) -> ConfidenceIntervalResult:
    r"""Confidence interval for a population variance, assuming normality.

    :math:`\left(\dfrac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}},
    \dfrac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}}\right)`, via
    :class:`scipy.stats.chi2`. See DeGroot & Schervish, *Probability and
    Statistics*, 4th ed., Sec. 9.8.

    Parameters
    ----------
    data : array-like, shape (n,)
    confidence_level : float

    Returns
    -------
    ConfidenceIntervalResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(loc=0.0, scale=3.0, size=200)
    >>> result = variance_confidence_interval(data)
    >>> bool(result.lower < 9.0 < result.upper)
    True
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    df = n - 1
    s2 = float(np.var(x, ddof=1))
    alpha = 1.0 - confidence_level
    chi2_upper = stats.chi2.ppf(1.0 - alpha / 2.0, df)
    chi2_lower = stats.chi2.ppf(alpha / 2.0, df)
    return ConfidenceIntervalResult(
        estimate=s2, lower=df * s2 / chi2_upper, upper=df * s2 / chi2_lower, confidence_level=confidence_level, method="chi_square_variance"
    )
