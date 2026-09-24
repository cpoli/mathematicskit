r"""Distribution-free and exact tests: Kolmogorov-Smirnov, Fisher's exact
test, the Wilcoxon signed-rank test, and the Mann-Whitney U test.

All wrap :mod:`scipy.stats` (``kstest``/``ks_2samp``, ``fisher_exact``,
``wilcoxon``, ``mannwhitneyu``), which compute exact small-sample
p-values where available. None of these tests assumes the data are
normal, which is the point of them. See Hollander, Wolfe & Chicken,
*Nonparametric Statistical Methods*, 3rd ed. (2014).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats

from mathematicskit.statistics.core.base import HypothesisTestResult

__all__ = ["kolmogorov_smirnov_test", "fisher_exact_test", "wilcoxon_signed_rank_test", "mann_whitney_u_test"]


def kolmogorov_smirnov_test(
    data: np.ndarray,
    reference: str | Callable | np.ndarray = "norm",
    args: tuple = (),
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    r"""Kolmogorov-Smirnov test against a distribution or a second sample.

    The statistic is the largest vertical gap between the empirical CDF
    :math:`F_n` and the reference CDF :math:`F`,
    :math:`D_n = \sup_x |F_n(x) - F(x)|`. Kolmogorov (1933) showed that
    :math:`\sqrt n D_n` has the same limiting distribution for *every*
    continuous :math:`F`; Smirnov (1939) extended it to two samples.
    Wraps :func:`scipy.stats.kstest` (one sample) or
    :func:`scipy.stats.ks_2samp` (two samples).

    Parameters
    ----------
    data : array-like, shape (n,)
    reference : str, callable, or array-like
        A ``scipy.stats`` distribution name or a CDF callable (one-sample
        test), or a second sample (two-sample test).
    args : tuple
        Parameters of the reference distribution, e.g. ``(loc, scale)``.
        They must be fixed in advance, not estimated from `data`.
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult

    Examples
    --------
    >>> import numpy as np
    >>> result = kolmogorov_smirnov_test(np.array([0.1, 0.4, 0.7]), "uniform")
    >>> round(result.statistic, 6)  # largest gap is at x = 0.7: 1 - 0.7
    0.3
    """
    x = np.asarray(data, dtype=np.float64)
    if isinstance(reference, str) or callable(reference):
        cdf = getattr(stats, reference)(*args).cdf if isinstance(reference, str) else reference
        res = stats.kstest(x, cdf, args=() if isinstance(reference, str) else args, alternative=alternative)
        method = "kolmogorov_smirnov"
    else:
        res = stats.ks_2samp(x, np.asarray(reference, dtype=np.float64), alternative=alternative)
        method = "kolmogorov_smirnov_2samp"
    return HypothesisTestResult(statistic=float(res.statistic), p_value=float(res.pvalue), method=method)


def fisher_exact_test(table: np.ndarray, alternative: str = "two-sided") -> HypothesisTestResult:
    r"""Fisher's exact test of independence for a 2x2 contingency table.

    Conditioning on both margins, the top-left count follows a
    hypergeometric distribution under independence, so the p-value is an
    exact sum of hypergeometric probabilities rather than a chi-square
    approximation. Wraps :func:`scipy.stats.fisher_exact`. See R. A.
    Fisher, *The Design of Experiments* (1935), Ch. 2.

    Parameters
    ----------
    table : array-like, shape (2, 2)
        Observed counts.
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult
        `statistic` is the sample odds ratio :math:`ad/bc`.

    Examples
    --------
    >>> # The lady tasting tea: all 8 cups classified correctly.
    >>> result = fisher_exact_test([[4, 0], [0, 4]], alternative="greater")
    >>> round(result.p_value * 70, 10)  # exactly 1/70
    1.0
    """
    t = np.asarray(table)
    if t.shape != (2, 2):
        raise ValueError("fisher_exact_test needs a 2x2 table")
    odds_ratio, p = stats.fisher_exact(t, alternative=alternative)
    return HypothesisTestResult(statistic=float(odds_ratio), p_value=float(p), method="fisher_exact")


def wilcoxon_signed_rank_test(x: np.ndarray, y: Optional[np.ndarray] = None, alternative: str = "two-sided") -> HypothesisTestResult:
    r"""Wilcoxon signed-rank test for a median of zero (or paired differences).

    Ranks the absolute differences :math:`|x_i - y_i|` and sums the ranks
    of the positive ones, :math:`W^+`. Under :math:`H_0` (differences
    symmetric about 0) each sign is a fair coin flip, which gives an
    exact null distribution. Wraps :func:`scipy.stats.wilcoxon`. See F.
    Wilcoxon, Biometrics Bulletin 1(6) (1945), 80-83.

    Parameters
    ----------
    x : array-like, shape (n,)
    y : array-like, shape (n,), optional
        Paired sample; if given, the test is applied to ``x - y``.
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult
        `statistic` is scipy's convention: :math:`\min(W^+, W^-)` for
        two-sided tests, :math:`W^+` otherwise.

    Examples
    --------
    >>> # 5 positive differences: exact one-sided p = (1/2)^5
    >>> result = wilcoxon_signed_rank_test([1.0, 2.0, 3.0, 4.0, 5.0], alternative="greater")
    >>> result.p_value
    0.03125
    """
    a = np.asarray(x, dtype=np.float64)
    d = a if y is None else a - np.asarray(y, dtype=np.float64)
    res = stats.wilcoxon(d, alternative=alternative)
    return HypothesisTestResult(statistic=float(res.statistic), p_value=float(res.pvalue), method="wilcoxon_signed_rank")


def mann_whitney_u_test(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> HypothesisTestResult:
    r"""Mann-Whitney U (Wilcoxon rank-sum) test for two independent samples.

    :math:`U` counts the pairs :math:`(x_i, y_j)` with :math:`x_i > y_j`
    (ties count one half); :math:`U/(n_1 n_2)` estimates
    :math:`P(X > Y)`. Wraps :func:`scipy.stats.mannwhitneyu`. See
    Wilcoxon (1945) and H. B. Mann & D. R. Whitney, Ann. Math. Statist.
    18(1) (1947), 50-60.

    Parameters
    ----------
    x, y : array-like
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult
        `statistic` is :math:`U` for `x`.

    Examples
    --------
    >>> result = mann_whitney_u_test([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])
    >>> result.statistic  # every x beats every y: U = 3 * 3
    9.0
    """
    res = stats.mannwhitneyu(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), alternative=alternative)
    return HypothesisTestResult(statistic=float(res.statistic), p_value=float(res.pvalue), method="mann_whitney_u")
