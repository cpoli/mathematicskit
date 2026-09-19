r"""Bootstrap resampling for confidence intervals and standard errors.

Built directly on :func:`scipy.stats.bootstrap`, which already
implements the percentile, "basic", and bias-corrected-and-accelerated
(BCa) bootstrap confidence-interval methods correctly (including BCa's
jackknife-based acceleration/bias correction, which is fiddly to get
right by hand) -- there is no reason to reimplement resampling-with-
replacement and interval construction from scratch. See Efron &
Tibshirani, *An Introduction to the Bootstrap*, 1993, Ch. 12-14.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

from mathkit.statistics.core.base import BootstrapResult

__all__ = ["bootstrap_confidence_interval"]


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    method: str = "BCa",
    seed: int = 0,
) -> BootstrapResult:
    r"""Bootstrap confidence interval and standard error for an arbitrary statistic.

    Resamples `data` with replacement `n_resamples` times, computes
    `statistic` on each resample, and builds a confidence interval from
    the resulting bootstrap distribution -- the standard nonparametric
    alternative to a closed-form interval (e.g.
    :func:`~mathkit.statistics.systems.confidence_intervals.mean_confidence_interval`)
    when no convenient parametric form is known, or as a check on one
    that is. See Efron & Tibshirani, *An Introduction to the Bootstrap*,
    1993, Ch. 12-14.

    Parameters
    ----------
    data : array-like, shape (n,)
    statistic : callable
        ``statistic(data) -> float``, computed along the last axis by
        ``scipy.stats.bootstrap`` internally (e.g. :func:`numpy.mean`,
        :func:`numpy.median`, or :func:`numpy.std`).
    n_resamples : int
    confidence_level : float
    method : {"percentile", "basic", "BCa"}
        Bootstrap CI construction method (see ``scipy.stats.bootstrap``).
    seed : int

    Returns
    -------
    BootstrapResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(loc=5.0, scale=2.0, size=200)
    >>> result = bootstrap_confidence_interval(data, statistic=np.mean, seed=0)
    >>> result.lower < 5.0 < result.upper
    True
    """
    x = np.asarray(data, dtype=np.float64)
    res = stats.bootstrap((x,), statistic, n_resamples=n_resamples, confidence_level=confidence_level, method=method, random_state=seed)
    return BootstrapResult(
        estimate=float(statistic(x)),
        lower=float(res.confidence_interval.low),
        upper=float(res.confidence_interval.high),
        std_error=float(res.standard_error),
        confidence_level=confidence_level,
        method=method,
    )
