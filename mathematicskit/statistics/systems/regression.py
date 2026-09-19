r"""Ordinary least-squares linear regression with residual diagnostics.

Fits via :func:`numpy.linalg.lstsq`, exactly as
:mod:`mathematicskit.numerical_analysis.systems.regression`'s polynomial
regression (this module adds the standard inferential diagnostics
around it: coefficient standard errors, t-statistics/p-values, and
R^2/adjusted R^2, referred to :class:`scipy.stats.t`). See DeGroot &
Schervish, *Probability and Statistics*, 4th ed., Ch. 11.5-11.7.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mathematicskit.statistics.core.base import RegressionResult

__all__ = ["linear_regression"]


def linear_regression(x: np.ndarray, y: np.ndarray, add_intercept: bool = True) -> RegressionResult:
    r"""Ordinary least squares: :math:`y = X\beta + \varepsilon`.

    Solves for :math:`\hat\beta` via :func:`numpy.linalg.lstsq`; standard
    errors come from the diagonal of :math:`\hat\sigma^2 (X^TX)^{-1}`
    where :math:`\hat\sigma^2 = SSR/(n-p)`, and each coefficient's
    t-statistic/p-value tests :math:`H_0: \beta_j = 0` against
    :class:`scipy.stats.t` with :math:`n-p` degrees of freedom. See
    DeGroot & Schervish, *Probability and Statistics*, 4th ed.,
    Sec. 11.5-11.7.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, k)
        Predictor(s). A 1D array is treated as a single predictor.
    y : array-like, shape (n,)
        Response.
    add_intercept : bool
        Whether to prepend a column of ones to `x` (fitting an
        intercept term). ``coefficients[0]`` is then the intercept.

    Returns
    -------
    RegressionResult

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = 2.0 * x + 1.0  # noiseless: fit should be essentially exact
    >>> result = linear_regression(x, y)
    >>> np.allclose(result.coefficients, [1.0, 2.0], atol=1e-8)
    True
    >>> round(result.r_squared, 6)
    1.0
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    design = np.column_stack([np.ones(n), x]) if add_intercept else x
    p = design.shape[1]

    coefficients, _residuals, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    df_resid = n - p
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / df_resid if df_resid > 0 and ss_tot > 0 else r_squared

    if df_resid > 0:
        sigma2 = ss_res / df_resid
        cov_beta = sigma2 * np.linalg.inv(design.T @ design)
        standard_errors = np.sqrt(np.diag(cov_beta))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_statistics = coefficients / standard_errors
        p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_statistics), df_resid))
    else:
        standard_errors = np.full(p, np.nan)
        t_statistics = np.full(p, np.nan)
        p_values = np.full(p, np.nan)

    return RegressionResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        t_statistics=t_statistics,
        p_values=p_values,
        fitted_values=fitted,
        residuals=residuals,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
    )
