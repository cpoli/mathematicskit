r"""Multiple-testing corrections: Bonferroni and Benjamini-Hochberg.

Hand-written: both are a few lines of sorting and cumulative minima,
and ``scipy.stats.false_discovery_control`` only exists from SciPy 1.11
(this package supports SciPy 1.10). Bonferroni controls the family-wise
error rate; Benjamini-Hochberg controls the false discovery rate
:math:`E[V/\max(R,1)]` for independent tests. See Y. Benjamini & Y.
Hochberg, J. R. Statist. Soc. B 57(1) (1995), 289-300.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.statistics.core.base import MultipleTestingResult

__all__ = ["bonferroni_correction", "benjamini_hochberg"]


def _as_p_values(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a nonempty 1D array with entries in [0, 1]")
    return p


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> MultipleTestingResult:
    r"""Bonferroni correction: reject :math:`H_i` when :math:`m p_i \le \alpha`.

    Controls the family-wise error rate (the chance of *any* false
    rejection) at :math:`\alpha`, under any dependence between tests.

    Parameters
    ----------
    p_values : array-like, shape (m,)
    alpha : float

    Returns
    -------
    MultipleTestingResult

    Examples
    --------
    >>> bonferroni_correction([0.01, 0.02, 0.2]).adjusted_p_values
    array([0.03, 0.06, 0.6 ])
    """
    p = _as_p_values(p_values)
    adjusted = np.minimum(p * p.size, 1.0)
    return MultipleTestingResult(rejected=adjusted <= alpha, adjusted_p_values=adjusted, alpha=alpha, method="bonferroni")


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> MultipleTestingResult:
    r"""Benjamini-Hochberg step-up procedure controlling the false discovery rate.

    Sort the p-values :math:`p_{(1)} \le \dots \le p_{(m)}`, find the
    largest :math:`k` with :math:`p_{(k)} \le k\alpha/m`, and reject
    :math:`H_{(1)}, \dots, H_{(k)}`. The adjusted p-values are
    :math:`\min_{j \ge i} \min(1, m p_{(j)}/j)`. See Benjamini & Hochberg
    (1995), J. R. Statist. Soc. B 57(1), 289-300.

    Parameters
    ----------
    p_values : array-like, shape (m,)
    alpha : float
        Target false discovery rate.

    Returns
    -------
    MultipleTestingResult

    Examples
    --------
    >>> result = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    >>> [round(float(v), 4) for v in result.adjusted_p_values]
    [0.04, 0.0533, 0.0533, 0.5]
    >>> result.rejected
    array([ True, False, False, False])
    """
    p = _as_p_values(p_values)
    m = p.size
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return MultipleTestingResult(rejected=adjusted <= alpha, adjusted_p_values=adjusted, alpha=alpha, method="benjamini_hochberg")
