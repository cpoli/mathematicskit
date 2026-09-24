r"""The jackknife: leave-one-out estimates of bias and standard error.

Hand-written because neither numpy nor scipy provides a general
jackknife (``scipy.stats.bootstrap`` uses one only internally, for the
BCa acceleration constant). With leave-one-out replicates
:math:`\hat\theta_{(i)}` and their mean :math:`\hat\theta_{(\cdot)}`,
Quenouille's bias estimate is
:math:`(n-1)(\hat\theta_{(\cdot)} - \hat\theta)` and Tukey's standard
error is :math:`\sqrt{\tfrac{n-1}{n}\sum_i(\hat\theta_{(i)} -
\hat\theta_{(\cdot)})^2}`. See M. H. Quenouille, Biometrika 43 (1956),
353-360; J. W. Tukey, Ann. Math. Statist. 29 (1958), 614; Efron &
Tibshirani, *An Introduction to the Bootstrap* (1993), Ch. 11.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathematicskit.statistics.core.base import JackknifeResult

__all__ = ["jackknife"]


def jackknife(data: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean) -> JackknifeResult:
    r"""Jackknife bias and standard-error estimates for a statistic.

    Recomputes `statistic` on each of the :math:`n` samples that leave
    one observation out. For the sample mean the jackknife standard
    error is exactly :math:`s/\sqrt n`; for the plug-in variance
    :math:`\frac1n\sum(x_i-\bar x)^2` the bias-corrected estimate is
    exactly the unbiased :math:`\frac1{n-1}\sum(x_i-\bar x)^2`.

    Parameters
    ----------
    data : array-like, shape (n,)
    statistic : callable
        ``statistic(sample) -> float``.

    Returns
    -------
    JackknifeResult

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([2.0, 4.0, 4.0, 5.0, 7.0, 9.0])
    >>> result = jackknife(data, np.var)  # plug-in variance, ddof=0
    >>> bool(np.isclose(result.bias_corrected, np.var(data, ddof=1)))
    True
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("jackknife needs at least 2 observations")
    estimate = float(statistic(x))
    replicates = np.array([float(statistic(np.delete(x, i))) for i in range(n)])
    mean_replicate = float(replicates.mean())
    bias = (n - 1) * (mean_replicate - estimate)
    std_error = float(np.sqrt((n - 1) / n * np.sum((replicates - mean_replicate) ** 2)))
    return JackknifeResult(estimate=estimate, bias=bias, std_error=std_error, bias_corrected=estimate - bias, replicates=replicates)
