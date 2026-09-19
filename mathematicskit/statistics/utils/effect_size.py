r"""Effect-size measures -- supporting numerics that accompany a
hypothesis test's p-value (a significant result can still be a tiny,
practically unimportant effect), not a test in their own right.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cohens_d"]


def cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    r"""Cohen's d: a standardized (unit-free) mean difference between two groups.

    :math:`d = \dfrac{\bar x_1 - \bar x_2}{s_{\text{pooled}}}`, where
    :math:`s_{\text{pooled}} = \sqrt{\dfrac{(n_1-1)s_1^2 +
    (n_2-1)s_2^2}{n_1+n_2-2}}`. Conventional (Cohen, 1988) rough
    benchmarks: :math:`|d|\approx 0.2` small, :math:`0.5` medium,
    :math:`0.8` large -- useful alongside
    :func:`~mathematicskit.statistics.systems.hypothesis_tests.two_sample_t_test`,
    since a test can be statistically significant (small p-value) while
    the underlying effect is practically negligible, especially at large
    sample sizes. See Cohen, *Statistical Power Analysis for the
    Behavioral Sciences*, 2nd ed., 1988, Ch. 2.

    Parameters
    ----------
    data1, data2 : array-like

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> b = a + 1.0  # every value shifted up by 1
    >>> round(cohens_d(a, b), 4)
    -0.6325
    """
    x1, x2 = np.asarray(data1, dtype=np.float64), np.asarray(data2, dtype=np.float64)
    n1, n2 = x1.shape[0], x2.shape[0]
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(x1) - np.mean(x2)) / pooled_std)
