r"""Descriptive statistics: central tendency, dispersion, shape, and order statistics.

Built directly on :mod:`numpy`/:mod:`scipy.stats` (``np.mean``,
``np.var``, ``np.percentile``, ``scipy.stats.skew``/``kurtosis``) --
these are exactly the well-known formulas every introductory statistics
text defines (e.g. DeGroot & Schervish, *Probability and Statistics*,
4th ed., Ch. 9.1-9.2), with no reason to recompute them by hand when the
library already does so correctly and efficiently.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mathematicskit.statistics.core.base import DescriptiveStatsResult

__all__ = ["descriptive_stats", "order_statistic"]


def descriptive_stats(data: np.ndarray) -> DescriptiveStatsResult:
    r"""Compute a standard descriptive-statistics summary of a dataset.

    Mean, variance/standard deviation (``ddof=1``, the unbiased sample
    estimator), Fisher-Pearson skewness, excess kurtosis, and the
    five-number summary (min, quartiles, max). See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 9.1-9.2.

    Parameters
    ----------
    data : array-like, shape (n,)

    Returns
    -------
    DescriptiveStatsResult

    Examples
    --------
    >>> result = descriptive_stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    >>> result.n
    8
    >>> result.mean
    5.0
    >>> round(result.std, 4)
    2.1381
    """
    x = np.asarray(data, dtype=np.float64)
    return DescriptiveStatsResult(
        n=x.shape[0],
        mean=float(np.mean(x)),
        variance=float(np.var(x, ddof=1)),
        std=float(np.std(x, ddof=1)),
        skewness=float(stats.skew(x)),
        kurtosis=float(stats.kurtosis(x)),
        minimum=float(np.min(x)),
        q1=float(np.percentile(x, 25)),
        median=float(np.median(x)),
        q3=float(np.percentile(x, 75)),
        maximum=float(np.max(x)),
    )


def order_statistic(data: np.ndarray, k: int) -> float:
    r"""The ``k``-th order statistic (1-indexed: ``k=1`` is the minimum).

    Parameters
    ----------
    data : array-like, shape (n,)
    k : int
        ``1 <= k <= n``.

    Returns
    -------
    float

    Examples
    --------
    >>> order_statistic([5.0, 1.0, 3.0, 2.0, 4.0], k=1)
    1.0
    >>> order_statistic([5.0, 1.0, 3.0, 2.0, 4.0], k=5)
    5.0
    """
    x = np.sort(np.asarray(data, dtype=np.float64))
    if not (1 <= k <= x.shape[0]):
        raise ValueError(f"k must be between 1 and {x.shape[0]}")
    return float(x[k - 1])
