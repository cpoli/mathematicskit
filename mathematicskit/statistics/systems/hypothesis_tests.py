r"""Hypothesis tests: one/two-sample z- and t-tests, chi-square
goodness-of-fit and independence tests, and one-way ANOVA.

The z-tests use :class:`mathematicskit.probability.Normal` for the p-value
(mathematicskit.probability's own reference distribution); the t-test,
chi-square tests, and ANOVA reference distributions (Student's t,
:math:`\chi^2`, F) aren't part of mathematicskit.probability's roster (which
covers binomial/Poisson/geometric/uniform/exponential/normal/gamma), so
their p-values come directly from :mod:`scipy.stats` (``scipy.stats.t``,
``.chi2``, ``.f``) -- consistent with the same "call the library
directly" rule. The test statistics themselves are simple closed-form
arithmetic, not something a library computes for you as a standalone
step. See DeGroot & Schervish, *Probability and Statistics*, 4th ed.,
Ch. 9 (estimation) and Ch. 10 (hypothesis testing).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mathematicskit.probability import Normal
from mathematicskit.statistics.core.base import HypothesisTestResult

__all__ = [
    "one_sample_z_test",
    "two_sample_z_test",
    "one_sample_t_test",
    "two_sample_t_test",
    "chi_square_goodness_of_fit",
    "chi_square_independence",
    "one_way_anova",
]

_STANDARD_NORMAL = Normal(mu=0.0, sigma=1.0)


def _p_value(stat: float, cdf, alternative: str) -> float:
    if alternative == "two-sided":
        return float(2.0 * min(cdf(stat), 1.0 - cdf(stat)))
    if alternative == "less":
        return float(cdf(stat))
    if alternative == "greater":
        return float(1.0 - cdf(stat))
    raise ValueError(f"alternative must be 'two-sided', 'less', or 'greater', got {alternative!r}")


def one_sample_z_test(data: np.ndarray, mu0: float, sigma: float, alternative: str = "two-sided") -> HypothesisTestResult:
    r"""One-sample z-test for the mean, with known population standard deviation.

    :math:`z = \dfrac{\bar x - \mu_0}{\sigma/\sqrt n}`, referred to
    :class:`mathematicskit.probability.Normal`\ (0, 1). See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 9.5.

    Parameters
    ----------
    data : array-like, shape (n,)
    mu0 : float
        Hypothesized mean under :math:`H_0`.
    sigma : float
        Known population standard deviation.
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(loc=6.0, scale=2.0, size=500)
    >>> result = one_sample_z_test(data, mu0=5.0, sigma=2.0)
    >>> bool(result.reject_null(alpha=0.05))
    True
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    z = (float(np.mean(x)) - mu0) / (sigma / np.sqrt(n))
    p = _p_value(z, _STANDARD_NORMAL.cdf, alternative)
    return HypothesisTestResult(statistic=z, p_value=p, method="one_sample_z")


def two_sample_z_test(data1: np.ndarray, data2: np.ndarray, sigma1: float, sigma2: float, alternative: str = "two-sided") -> HypothesisTestResult:
    r"""Two-sample z-test for a difference of means, with known population standard deviations.

    :math:`z = \dfrac{\bar x_1 - \bar x_2}{\sqrt{\sigma_1^2/n_1 +
    \sigma_2^2/n_2}}`. See DeGroot & Schervish, *Probability and
    Statistics*, 4th ed., Sec. 9.6.

    Parameters
    ----------
    data1, data2 : array-like
    sigma1, sigma2 : float
        Known population standard deviations.
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult
    """
    x1, x2 = np.asarray(data1, dtype=np.float64), np.asarray(data2, dtype=np.float64)
    n1, n2 = x1.shape[0], x2.shape[0]
    se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
    z = (float(np.mean(x1)) - float(np.mean(x2))) / se
    p = _p_value(z, _STANDARD_NORMAL.cdf, alternative)
    return HypothesisTestResult(statistic=z, p_value=p, method="two_sample_z")


def one_sample_t_test(data: np.ndarray, mu0: float, alternative: str = "two-sided") -> HypothesisTestResult:
    r"""One-sample t-test for the mean, with unknown population standard deviation.

    :math:`t = \dfrac{\bar x - \mu_0}{s/\sqrt n}`, referred to Student's
    t distribution with :math:`n-1` degrees of freedom
    (:class:`scipy.stats.t`). See DeGroot & Schervish, *Probability and
    Statistics*, 4th ed., Sec. 9.5.

    Parameters
    ----------
    data : array-like, shape (n,)
    mu0 : float
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> data = rng.normal(loc=100.0, scale=15.0, size=40)
    >>> result = one_sample_t_test(data, mu0=100.0)
    >>> result.df
    39.0
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    df = float(n - 1)
    t = (float(np.mean(x)) - mu0) / (float(np.std(x, ddof=1)) / np.sqrt(n))
    p = _p_value(t, lambda v: stats.t.cdf(v, df), alternative)
    return HypothesisTestResult(statistic=t, p_value=p, df=df, method="one_sample_t")


def two_sample_t_test(data1: np.ndarray, data2: np.ndarray, equal_var: bool = True, alternative: str = "two-sided") -> HypothesisTestResult:
    r"""Two-sample t-test for a difference of means.

    Pooled-variance ("Student's") t-test if ``equal_var=True``,
    otherwise Welch's t-test with the Welch-Satterthwaite degrees of
    freedom. Referred to :class:`scipy.stats.t`. See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 9.6, and Welch (1947),
    Biometrika 34.

    Parameters
    ----------
    data1, data2 : array-like
    equal_var : bool
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    HypothesisTestResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> a = rng.normal(loc=10.0, scale=2.0, size=50)
    >>> b = rng.normal(loc=11.0, scale=2.0, size=50)
    >>> result = two_sample_t_test(a, b)
    >>> bool(result.statistic < 0)
    True
    """
    x1, x2 = np.asarray(data1, dtype=np.float64), np.asarray(data2, dtype=np.float64)
    n1, n2 = x1.shape[0], x2.shape[0]
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    mean_diff = float(np.mean(x1) - np.mean(x2))
    if equal_var:
        df = float(n1 + n2 - 2)
        pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = np.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t = mean_diff / se
    p = _p_value(t, lambda v: stats.t.cdf(v, df), alternative)
    return HypothesisTestResult(statistic=t, p_value=p, df=float(df), method="two_sample_t" if equal_var else "welch_t")


def chi_square_goodness_of_fit(observed: np.ndarray, expected: np.ndarray) -> HypothesisTestResult:
    r"""Chi-square goodness-of-fit test: does `observed` match `expected` proportions/counts?

    :math:`\chi^2 = \sum_i (O_i - E_i)^2/E_i`, referred to
    :class:`scipy.stats.chi2` with :math:`k-1` degrees of freedom
    (``k`` categories).

    The single degree of freedom subtracted is the one used up by the
    constraint that the counts sum to :math:`n`; this function assumes
    `expected` is fully specified in advance. If the expected counts were
    themselves obtained by estimating :math:`m` parameters from the same
    data (fitting a Poisson rate, say), the correct degrees of freedom are
    :math:`k-1-m`, and the p-value reported here will be too large --
    conservative about rejecting :math:`H_0`. See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 10.3.

    Parameters
    ----------
    observed : array-like, shape (k,)
        Observed counts.
    expected : array-like, shape (k,)
        Expected counts under :math:`H_0` (need not be pre-normalized to
        sum to the same total as `observed`, but should be positive).

    Returns
    -------
    HypothesisTestResult

    Examples
    --------
    >>> import numpy as np
    >>> # A fair six-sided die rolled 600 times: expect 100 of each face.
    >>> observed = np.array([90.0, 105.0, 95.0, 100.0, 110.0, 100.0])
    >>> result = chi_square_goodness_of_fit(observed, expected=np.full(6, 100.0))
    >>> result.df
    5.0
    >>> result.reject_null(alpha=0.05)
    False
    """
    o = np.asarray(observed, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    chi2 = float(np.sum((o - e) ** 2 / e))
    df = float(o.shape[0] - 1)
    p = float(1.0 - stats.chi2.cdf(chi2, df))
    return HypothesisTestResult(statistic=chi2, p_value=p, df=df, method="chi_square_goodness_of_fit")


def chi_square_independence(contingency_table: np.ndarray) -> HypothesisTestResult:
    r"""Chi-square test of independence for a two-way contingency table.

    Expected counts under independence are :math:`E_{ij} = (\text{row
    total}_i)(\text{col total}_j)/N`; :math:`\chi^2 = \sum_{ij}
    (O_{ij}-E_{ij})^2/E_{ij}`, referred to :class:`scipy.stats.chi2` with
    :math:`(r-1)(c-1)` degrees of freedom. See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 10.4.

    Parameters
    ----------
    contingency_table : array-like, shape (r, c)
        Observed counts.

    Returns
    -------
    HypothesisTestResult
        ``extra["expected"]`` holds the expected-counts table.

    Examples
    --------
    >>> import numpy as np
    >>> table = np.array([[30.0, 10.0], [20.0, 40.0]])
    >>> result = chi_square_independence(table)
    >>> result.df
    1.0
    >>> result.reject_null(alpha=0.01)
    True
    """
    o = np.asarray(contingency_table, dtype=np.float64)
    row_totals = o.sum(axis=1, keepdims=True)
    col_totals = o.sum(axis=0, keepdims=True)
    n = o.sum()
    e = row_totals @ col_totals / n
    chi2 = float(np.sum((o - e) ** 2 / e))
    df = float((o.shape[0] - 1) * (o.shape[1] - 1))
    p = float(1.0 - stats.chi2.cdf(chi2, df))
    return HypothesisTestResult(statistic=chi2, p_value=p, df=df, method="chi_square_independence", extra={"expected": e})


def one_way_anova(*groups: np.ndarray) -> HypothesisTestResult:
    r"""One-way analysis of variance: do 2+ groups share a common mean?

    :math:`F = \dfrac{\text{between-group variance}}{\text{within-group
    variance}} = \dfrac{SSB/(k-1)}{SSW/(N-k)}`, referred to
    :class:`scipy.stats.f` with :math:`(k-1, N-k)` degrees of freedom.
    See DeGroot & Schervish, *Probability and Statistics*, 4th ed.,
    Sec. 11.3.

    Parameters
    ----------
    *groups : array-like
        Two or more samples (one per group).

    Returns
    -------
    HypothesisTestResult
        ``extra["group_means"]`` holds each group's sample mean.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([4.0, 5.0, 6.0, 5.0])
    >>> b = np.array([7.0, 8.0, 7.0, 9.0])
    >>> c = np.array([4.0, 3.0, 5.0, 4.0])
    >>> result = one_way_anova(a, b, c)
    >>> result.reject_null(alpha=0.01)
    True
    """
    if len(groups) < 2:
        raise ValueError("one_way_anova needs at least 2 groups")
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    k = len(arrays)
    n_total = sum(a.shape[0] for a in arrays)
    grand_mean = np.concatenate(arrays).mean()
    ssb = sum(a.shape[0] * (a.mean() - grand_mean) ** 2 for a in arrays)
    ssw = sum(np.sum((a - a.mean()) ** 2) for a in arrays)
    df_between, df_within = float(k - 1), float(n_total - k)
    f_stat = (ssb / df_between) / (ssw / df_within)
    p = float(1.0 - stats.f.cdf(f_stat, df_between, df_within))
    group_means = np.array([a.mean() for a in arrays])
    return HypothesisTestResult(
        statistic=float(f_stat), p_value=p, df=df_between, method="one_way_anova", extra={"df_within": df_within, "group_means": group_means}
    )
