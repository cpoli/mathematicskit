r"""Stochastic differential equations: the Euler-Maruyama scheme, geometric
Brownian motion, and the Ornstein-Uhlenbeck process.

An Itô SDE

.. math::

   dX_t = a(X_t, t)\,dt + b(X_t, t)\,dW_t

is defined by Itô's stochastic integral (1944). Maruyama (1955) showed
that the Euler step

.. math::

   X_{n+1} = X_n + a(X_n, t_n)\,\Delta t + b(X_n, t_n)\,\Delta W_n,
   \qquad \Delta W_n \sim \mathcal N(0, \Delta t),

converges to it: with strong order 1/2 (pathwise error
:math:`O(\Delta t^{1/2})`) and weak order 1 (error in expectations
:math:`O(\Delta t)`). Hand-rolled beside
:func:`~mathematicskit.probability.systems.stochastic_processes.brownian_motion`:
scipy has no SDE solver. See K. Itô, "Stochastic Integral," Proceedings
of the Imperial Academy 20(8) (1944), 519-524; G. Maruyama, "Continuous
Markov Processes and Stochastic Equations," Rendiconti del Circolo
Matematico di Palermo 4 (1955), 48-90; and D. J. Higham, "An Algorithmic
Introduction to Numerical Simulation of Stochastic Differential
Equations," SIAM Review 43(3) (2001), 525-546.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

from mathematicskit.probability.core.base import SDEResult

__all__ = ["euler_maruyama", "geometric_brownian_motion", "ornstein_uhlenbeck"]


def _wiener_increments(n_paths: int, n_steps: int, dt: float, seed: int) -> np.ndarray:
    return np.sqrt(dt) * np.random.default_rng(seed).standard_normal(size=(n_paths, n_steps))


def _wiener_paths(dw: np.ndarray) -> np.ndarray:
    w = np.zeros((dw.shape[0], dw.shape[1] + 1))
    np.cumsum(dw, axis=1, out=w[:, 1:])
    return w


def euler_maruyama(
    drift: Callable[[np.ndarray, float], np.ndarray],
    diffusion: Callable[[np.ndarray, float], np.ndarray],
    x0: float,
    t_max: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    seed: int = 0,
    dw: Optional[np.ndarray] = None,
) -> SDEResult:
    r"""Integrate the scalar Itô SDE :math:`dX = a(X, t)\,dt + b(X, t)\,dW` by the Euler-Maruyama scheme.

    Parameters
    ----------
    drift : callable
        ``drift(x, t) -> ndarray``, vectorized over the array of paths ``x``.
    diffusion : callable
        ``diffusion(x, t) -> ndarray``, vectorized likewise.
    x0 : float
        Initial value, shared by every path.
    t_max : float
    n_steps : int
    n_paths : int
    seed : int
        Seeds the Brownian increments (ignored when ``dw`` is given).
    dw : ndarray, shape (n_paths, n_steps), optional
        Brownian increments to use instead of fresh ones, for example a
        coarsened copy of a finer path's increments, so that two step
        sizes can be compared on the same Brownian path.

    Returns
    -------
    SDEResult

    Examples
    --------
    >>> import numpy as np
    >>> # dX = -X dt + dW from X(0) = 0 has Var X(t) = (1 - exp(-2t)) / 2.
    >>> result = euler_maruyama(lambda x, t: -x, lambda x, t: np.ones_like(x), 0.0, t_max=1.0, n_steps=200, n_paths=20000)
    >>> bool(abs(result.paths[:, -1].var() - (1 - np.exp(-2.0)) / 2) < 0.02)
    True
    """
    dt = t_max / n_steps
    if dw is None:
        dw = _wiener_increments(n_paths, n_steps, dt, seed)
    dw = np.asarray(dw, dtype=np.float64)
    n_paths = dw.shape[0]
    times = np.linspace(0.0, t_max, n_steps + 1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for n in range(n_steps):
        x, t = paths[:, n], times[n]
        paths[:, n + 1] = x + drift(x, t) * dt + diffusion(x, t) * dw[:, n]
    return SDEResult(times=times, paths=paths, wiener=_wiener_paths(dw), method="euler_maruyama")


def geometric_brownian_motion(
    mu: float,
    sigma: float,
    x0: float = 1.0,
    t_max: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    seed: int = 0,
    method: str = "exact",
) -> SDEResult:
    r"""Sample geometric Brownian motion :math:`dX = \mu X\,dt + \sigma X\,dW`.

    Itô's formula applied to :math:`\log X` gives the exact solution

    .. math::

       X_t = X_0 \exp\!\left(\left(\mu - \tfrac12\sigma^2\right)t + \sigma W_t\right),

    whose :math:`-\tfrac12\sigma^2` is the Itô correction that ordinary
    calculus would miss. ``method="euler_maruyama"`` integrates the SDE
    instead; with the same ``seed`` both methods use the same Brownian
    path, so their difference is the scheme's pathwise error.

    Parameters
    ----------
    mu, sigma : float
        Drift and volatility.
    x0 : float
    t_max : float
    n_steps : int
    n_paths : int
    seed : int
    method : {"exact", "euler_maruyama"}

    Returns
    -------
    SDEResult

    Examples
    --------
    >>> import numpy as np
    >>> result = geometric_brownian_motion(0.1, 0.4, n_steps=50, n_paths=40000, seed=0)
    >>> bool(abs(result.paths[:, -1].mean() - np.exp(0.1)) < 0.01)  # E X_t = X_0 e^{mu t}
    True
    """
    if method == "euler_maruyama":
        return euler_maruyama(lambda x, t: mu * x, lambda x, t: sigma * x, x0, t_max, n_steps, n_paths, seed)
    if method != "exact":
        raise ValueError(f"unknown method {method!r}")
    dw = _wiener_increments(n_paths, n_steps, t_max / n_steps, seed)
    times = np.linspace(0.0, t_max, n_steps + 1)
    wiener = _wiener_paths(dw)
    paths = x0 * np.exp((mu - 0.5 * sigma**2) * times + sigma * wiener)
    return SDEResult(times=times, paths=paths, wiener=wiener, method="exact")


def ornstein_uhlenbeck(
    theta: float,
    mu: float,
    sigma: float,
    x0: float = 0.0,
    t_max: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    seed: int = 0,
) -> SDEResult:
    r"""Sample the Ornstein-Uhlenbeck process :math:`dX = \theta(\mu - X)\,dt + \sigma\,dW` by Euler-Maruyama.

    The process reverts to :math:`\mu` at rate :math:`\theta`. It is
    Gaussian, with

    .. math::

       E X_t = \mu + (X_0 - \mu)e^{-\theta t}, \qquad
       \operatorname{Var} X_t = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta t}\right),

    and stationary distribution :math:`\mathcal N(\mu, \sigma^2/2\theta)`
    (Uhlenbeck and Ornstein, 1930).

    Parameters
    ----------
    theta : float
        Mean-reversion rate, ``theta > 0``.
    mu : float
        Long-run mean.
    sigma : float
        Noise amplitude.
    x0 : float
    t_max : float
    n_steps : int
    n_paths : int
    seed : int

    Returns
    -------
    SDEResult

    Examples
    --------
    >>> result = ornstein_uhlenbeck(2.0, 1.0, 0.5, x0=3.0, t_max=5.0, n_steps=500, n_paths=20000)
    >>> bool(abs(result.paths[:, -1].mean() - 1.0) < 0.01), bool(abs(result.paths[:, -1].var() - 0.0625) < 0.005)
    (True, True)
    """
    return euler_maruyama(lambda x, t: theta * (mu - x), lambda x, t: np.full_like(x, sigma), x0, t_max, n_steps, n_paths, seed)
