r"""Correlation coefficients: Pearson's product-moment correlation and
Spearman's rank correlation.

Both wrap :mod:`scipy.stats` (``pearsonr``, ``spearmanr``), which already
compute the coefficients and their p-values correctly. Pearson's
:math:`r` measures *linear* association; Spearman's :math:`\rho` is
Pearson's :math:`r` applied to the ranks, so it measures any
*monotone* association and is insensitive to outliers. See K. Pearson,
Phil. Trans. R. Soc. A 187 (1896), and C. Spearman, Am. J. Psychol. 15
(1904).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mathematicskit.statistics.core.base import CorrelationResult

__all__ = ["pearson_correlation", "spearman_correlation"]


def _paired(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("x and y must be 1D arrays of equal length")
    if a.shape[0] < 3:
        raise ValueError("need at least 3 paired observations")
    return a, b


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    r"""Pearson's product-moment correlation coefficient.

    .. math::

       r = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}
                {\sqrt{\sum_i (x_i - \bar x)^2 \sum_i (y_i - \bar y)^2}},

    with the p-value from :math:`t = r\sqrt{(n-2)/(1-r^2)}` referred to
    Student's t with :math:`n-2` degrees of freedom (exact for bivariate
    normal data). Wraps :func:`scipy.stats.pearsonr`. See Pearson (1896),
    Phil. Trans. R. Soc. A 187, 253-318.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Paired observations, :math:`n \ge 3`.

    Returns
    -------
    CorrelationResult

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = pearson_correlation(x, 2.0 * x + 1.0)
    >>> round(result.coefficient, 12)
    1.0
    """
    a, b = _paired(x, y)
    res = stats.pearsonr(a, b)
    return CorrelationResult(coefficient=float(res[0]), p_value=float(res[1]), n=int(a.shape[0]), method="pearson")


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    r"""Spearman's rank correlation coefficient.

    Pearson's :math:`r` computed on the ranks of `x` and `y` (ties get
    their average rank). Without ties it reduces to Spearman's formula

    .. math::

       \rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)},

    where :math:`d_i` is the difference between the two ranks of
    observation :math:`i`. Wraps :func:`scipy.stats.spearmanr`. See
    Spearman (1904), Am. J. Psychol. 15(1), 72-101.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Paired observations, :math:`n \ge 3`.

    Returns
    -------
    CorrelationResult

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> round(spearman_correlation(x, np.exp(x)).coefficient, 12)  # any increasing map gives 1
    1.0
    """
    a, b = _paired(x, y)
    res = stats.spearmanr(a, b)
    return CorrelationResult(coefficient=float(res[0]), p_value=float(res[1]), n=int(a.shape[0]), method="spearman")
