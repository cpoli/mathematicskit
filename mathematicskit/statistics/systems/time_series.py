r"""Time series: sample autocorrelation and autoregressive models.

Yule (1927) modelled Wolfer's sunspot numbers as a pendulum kicked by
random shocks, :math:`X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} +
\varepsilon_t`: the first autoregressive model. Multiplying an AR(p)
model by :math:`X_{t-k}` and taking expectations gives the Yule-Walker
equations

.. math::

   \rho_k = \sum_{j=1}^{p} \phi_j\,\rho_{|k-j|}, \qquad k = 1, \dots, p,

a symmetric Toeplitz system in the autocorrelations, solved here by
Levinson recursion via :func:`scipy.linalg.solve_toeplitz`. See G. U.
Yule, "On a Method of Investigating Periodicities in Disturbed Series,
with Special Reference to Wolfer's Sunspot Numbers," Philosophical
Transactions of the Royal Society A 226 (1927), 267-298; G. Walker, "On
Periodicity in Series of Related Terms," Proceedings of the Royal
Society A 131 (1931), 518-532; and Brockwell & Davis, *Introduction to
Time Series and Forecasting*, 3rd ed. (2016), Sec. 2.4 and 5.1.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, signal

from mathematicskit.statistics.core.base import ARModelResult

__all__ = ["autocorrelation", "yule_walker", "simulate_ar", "ar_autocorrelation"]


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    r"""Sample autocorrelation function :math:`\hat\rho_k`, ``k = 0..max_lag``.

    Uses the standard biased estimator
    :math:`\hat\gamma_k = \frac1n \sum_{t=1}^{n-k} (x_t - \bar x)(x_{t+k} - \bar x)`,
    :math:`\hat\rho_k = \hat\gamma_k/\hat\gamma_0`, whose divisor :math:`n`
    (not :math:`n - k`) keeps the autocovariance matrix positive
    semidefinite, so the Yule-Walker system is always solvable.

    Parameters
    ----------
    x : array-like, shape (n,)
    max_lag : int

    Returns
    -------
    ndarray, shape (max_lag + 1,)

    Examples
    --------
    >>> autocorrelation([1.0, -1.0, 1.0, -1.0], 2)
    array([ 1.  , -0.75,  0.5 ])
    """
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean()
    n = centered.size
    full = np.correlate(centered, centered, mode="full")[n - 1 : n + max_lag]
    return full / full[0]


def yule_walker(x: np.ndarray, order: int) -> ARModelResult:
    r"""Fit an AR(``order``) model by solving the Yule-Walker equations.

    Parameters
    ----------
    x : array-like, shape (n,)
    order : int
        The autoregressive order :math:`p \geq 1`.

    Returns
    -------
    ARModelResult
        The innovation variance is
        :math:`\hat\sigma^2 = \hat\gamma_0 (1 - \sum_k \hat\phi_k \hat\rho_k)`.

    Examples
    --------
    >>> x = simulate_ar([0.5, -0.3], n=50000, seed=0)
    >>> result = yule_walker(x, 2)
    >>> np.round(result.coefficients, 1)
    array([ 0.5, -0.3])
    """
    x = np.asarray(x, dtype=np.float64)
    rho = autocorrelation(x, order)
    phi = linalg.solve_toeplitz(rho[:-1], rho[1:])
    gamma0 = float(np.mean((x - x.mean()) ** 2))
    return ARModelResult(coefficients=phi, noise_variance=gamma0 * float(1.0 - phi @ rho[1:]), mean=float(x.mean()), autocorrelation=rho)


def simulate_ar(coefficients, n: int, sigma: float = 1.0, mean: float = 0.0, burn_in: int = 500, seed: int = 0) -> np.ndarray:
    r"""Simulate :math:`X_t - \mu = \sum_k \phi_k (X_{t-k} - \mu) + \varepsilon_t`, :math:`\varepsilon_t \sim \mathcal N(0, \sigma^2)`.

    The recursion is run by :func:`scipy.signal.lfilter`, and the first
    ``burn_in`` values are discarded so the returned series is close to
    stationary.

    Parameters
    ----------
    coefficients : array-like, shape (p,)
    n : int
    sigma : float
    mean : float
    burn_in : int
    seed : int

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> x = simulate_ar([0.9], n=100000, seed=1)
    >>> bool(abs(x.var() - 1 / (1 - 0.81)) < 0.2)  # Var = sigma^2 / (1 - phi^2)
    True
    """
    phi = np.asarray(coefficients, dtype=np.float64)
    noise = sigma * np.random.default_rng(seed).standard_normal(n + burn_in)
    return mean + signal.lfilter([1.0], np.concatenate(([1.0], -phi)), noise)[burn_in:]


def ar_autocorrelation(coefficients, max_lag: int) -> np.ndarray:
    r"""Theoretical autocorrelation of a stationary AR(p) process.

    Solves the Yule-Walker equations the other way round for
    :math:`\rho_1, \dots, \rho_p`, then extends them by the recursion
    :math:`\rho_k = \sum_j \phi_j \rho_{k-j}`.

    Parameters
    ----------
    coefficients : array-like, shape (p,)
    max_lag : int

    Returns
    -------
    ndarray, shape (max_lag + 1,)

    Examples
    --------
    >>> ar_autocorrelation([0.5], 3)  # AR(1): rho_k = phi^k
    array([1.   , 0.5  , 0.25 , 0.125])
    """
    phi = np.asarray(coefficients, dtype=np.float64)
    p = phi.size
    # rho_k - sum_j phi_j rho_{|k-j|} = 0 for k = 1..p, with rho_0 = 1, as a linear system in rho_1..rho_p.
    a = np.eye(p)
    b = np.zeros(p)
    for k in range(1, p + 1):
        for j in range(1, p + 1):
            lag = abs(k - j)
            if lag == 0:
                b[k - 1] += phi[j - 1]
            else:
                a[k - 1, lag - 1] -= phi[j - 1]
    rho = np.ones(max(max_lag, p) + 1)
    rho[1 : p + 1] = np.linalg.solve(a, b)
    for k in range(p + 1, max_lag + 1):
        rho[k] = phi @ rho[k - p : k][::-1]
    return rho[: max_lag + 1]
