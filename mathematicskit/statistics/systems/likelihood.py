r"""Maximum-likelihood estimation and the likelihood-ratio test.

:func:`maximum_likelihood_fit` wraps the ``fit`` method of any
continuous :mod:`scipy.stats` distribution, which maximizes the
log-likelihood :math:`\ell(\theta) = \sum_i \log f(x_i; \theta)`
(R. A. Fisher, Phil. Trans. R. Soc. A 222 (1922), 309-368).
:func:`likelihood_ratio_test` refers :math:`2(\ell_1 - \ell_0)` to the
chi-square distribution, as justified by Wilks's theorem (S. S. Wilks,
Ann. Math. Statist. 9(1) (1938), 60-62).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mathematicskit.statistics.core.base import HypothesisTestResult, MaximumLikelihoodResult

__all__ = ["maximum_likelihood_fit", "likelihood_ratio_test"]


def maximum_likelihood_fit(data: np.ndarray, distribution: str | stats.rv_continuous = "norm", **fixed) -> MaximumLikelihoodResult:
    r"""Fit a continuous distribution to `data` by maximum likelihood.

    Finds :math:`\hat\theta = \arg\max_\theta \sum_i \log f(x_i;\theta)`
    using ``scipy.stats.<distribution>.fit``. For the normal distribution
    the maximizer is closed-form: :math:`\hat\mu = \bar x` and
    :math:`\hat\sigma^2 = \frac1n\sum_i (x_i-\bar x)^2` (note the
    :math:`1/n`, not :math:`1/(n-1)`). See Fisher (1922), Phil. Trans. R.
    Soc. A 222, 309-368.

    Parameters
    ----------
    data : array-like, shape (n,)
    distribution : str or scipy.stats.rv_continuous
        A continuous ``scipy.stats`` distribution or its name
        (``"norm"``, ``"expon"``, ``"gamma"``, ...).
    **fixed
        Parameters held fixed rather than estimated, using ``scipy``'s
        ``f``-prefix convention (e.g. ``floc=0``).

    Returns
    -------
    MaximumLikelihoodResult

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = maximum_likelihood_fit(data, "norm")
    >>> [round(float(v), 6) for v in result.params]  # mean and sqrt(2)
    [3.0, 1.414214]
    """
    x = np.asarray(data, dtype=np.float64)
    dist = getattr(stats, distribution) if isinstance(distribution, str) else distribution
    params = tuple(float(v) for v in dist.fit(x, **fixed))
    log_likelihood = float(np.sum(dist.logpdf(x, *params)))
    name = distribution if isinstance(distribution, str) else getattr(dist, "name", "")
    return MaximumLikelihoodResult(params=params, log_likelihood=log_likelihood, n_params=len(params) - len(fixed), distribution=name)


def likelihood_ratio_test(log_likelihood_null: float, log_likelihood_alt: float, df: int) -> HypothesisTestResult:
    r"""Likelihood-ratio test of a nested null model against a larger model.

    The statistic :math:`\Lambda = 2(\ell_1 - \ell_0)` is, by Wilks's
    theorem, asymptotically :math:`\chi^2_{df}` under :math:`H_0`, where
    `df` is the number of extra free parameters in the alternative model.
    See Wilks (1938), Ann. Math. Statist. 9(1), 60-62.

    Parameters
    ----------
    log_likelihood_null : float
        Maximized log-likelihood of the restricted (null) model.
    log_likelihood_alt : float
        Maximized log-likelihood of the full (alternative) model.
    df : int
        Difference in the number of free parameters.

    Returns
    -------
    HypothesisTestResult

    Examples
    --------
    >>> result = likelihood_ratio_test(-10.0, -8.0, df=1)
    >>> result.statistic
    4.0
    >>> round(result.p_value, 4)
    0.0455
    """
    if df < 1:
        raise ValueError("df must be a positive integer")
    lam = 2.0 * (float(log_likelihood_alt) - float(log_likelihood_null))
    p = float(stats.chi2.sf(lam, df))
    return HypothesisTestResult(statistic=lam, p_value=p, df=float(df), method="likelihood_ratio")
