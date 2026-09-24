r"""Population models: Verhulst's logistic growth and the Lotka-Volterra
predator-prey system.

See Verhulst (1838), Lotka (1920), Volterra (1926), and Strogatz,
*Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 2.3 (logistic equation) and
Ch. 6.4-6.5 (predator-prey models and conserved quantities).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from mathematicskit.ode_dynamics.core.base import FlowSystem

__all__ = ["LogisticGrowth", "logistic_growth_solution", "LotkaVolterra", "lotka_volterra_invariant"]


@lru_cache(maxsize=64)
def _make_logistic_rhs(r: float, K: float):
    @njit(cache=True)
    def rhs(state, t, params):
        out = np.empty(1)
        out[0] = r * state[0] * (1.0 - state[0] / K)
        return out

    return rhs


class LogisticGrowth(FlowSystem):
    r"""Verhulst's logistic equation :math:`\dot N = rN(1 - N/K)`.

    Growth is exponential while :math:`N \ll K` and saturates at the
    carrying capacity :math:`K`, a stable fixed point; :math:`N=0` is
    unstable. See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed.,
    Ch. 2.3.

    Parameters
    ----------
    state0 : array-like, shape (1,)
        Initial population ``[N0]``.
    r : float
        Intrinsic growth rate.
    K : float
        Carrying capacity.

    Examples
    --------
    >>> system = LogisticGrowth([10.0], r=1.0, K=100.0)
    >>> result = system.integrate((0.0, 20.0), dt=1e-2, method="rk4")
    >>> round(float(result.y[-1, 0]), 3)
    100.0
    """

    def __init__(self, state0, r: float = 1.0, K: float = 1.0):
        self.r, self.K = float(r), float(K)
        self._rhs_njit = _make_logistic_rhs(self.r, self.K)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        return np.array([self.r * state[0] * (1.0 - state[0] / self.K)])


def logistic_growth_solution(t, n0: float, r: float, K: float) -> np.ndarray:
    r"""Closed-form solution of the logistic equation.

    .. math:: N(t) = \frac{K}{1 + \left(\frac{K - N_0}{N_0}\right) e^{-rt}}

    (Verhulst, 1838), the sigmoid "logistic curve".

    Parameters
    ----------
    t : float or array-like
        Time(s).
    n0 : float
        Initial population (``> 0``).
    r, K : float
        Growth rate and carrying capacity.

    Returns
    -------
    ndarray
        :math:`N(t)`.

    Examples
    --------
    >>> float(logistic_growth_solution(0.0, n0=10.0, r=1.0, K=100.0))
    10.0
    >>> round(float(logistic_growth_solution(50.0, n0=10.0, r=1.0, K=100.0)), 6)
    100.0
    """
    if n0 <= 0:
        raise ValueError("n0 must be positive")
    t = np.asarray(t, dtype=np.float64)
    return K / (1.0 + (K - n0) / n0 * np.exp(-r * t))


@lru_cache(maxsize=64)
def _make_lotka_volterra_rhs(alpha: float, beta: float, delta: float, gamma: float):
    @njit(cache=True)
    def rhs(state, t, params):
        x, y = state[0], state[1]
        out = np.empty(2)
        out[0] = alpha * x - beta * x * y
        out[1] = delta * x * y - gamma * y
        return out

    return rhs


class LotkaVolterra(FlowSystem):
    r"""Lotka-Volterra predator-prey system.

    :math:`\dot x = \alpha x - \beta xy` (prey), :math:`\dot y = \delta
    xy - \gamma y` (predators). The coexistence fixed point
    :math:`(\gamma/\delta,\ \alpha/\beta)` is a nonlinear center: every
    orbit around it is closed, because the quantity returned by
    :func:`lotka_volterra_invariant` is conserved. Small oscillations
    have period :math:`2\pi/\sqrt{\alpha\gamma}`. See Strogatz,
    *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 6.4-6.5.

    Parameters
    ----------
    state0 : array-like, shape (2,)
        Initial ``(prey, predators)``.
    alpha, beta, delta, gamma : float
        Prey growth, predation, predator conversion, and predator death
        rates.

    Examples
    --------
    >>> system = LotkaVolterra([1.0, 0.5], alpha=1.0, beta=1.0, delta=1.0, gamma=1.0)
    >>> result = system.integrate((0.0, 10.0), dt=1e-3, method="rk4")
    >>> result.y.shape[1]
    2
    >>> system.fixed_point()
    array([1., 1.])
    """

    def __init__(self, state0, alpha: float = 1.0, beta: float = 1.0, delta: float = 1.0, gamma: float = 1.0):
        self.alpha, self.beta, self.delta, self.gamma = float(alpha), float(beta), float(delta), float(gamma)
        self._rhs_njit = _make_lotka_volterra_rhs(self.alpha, self.beta, self.delta, self.gamma)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        x, y = state[0], state[1]
        return np.array([self.alpha * x - self.beta * x * y, self.delta * x * y - self.gamma * y])

    def fixed_point(self) -> np.ndarray:
        """Coexistence fixed point ``(gamma/delta, alpha/beta)``."""
        return np.array([self.gamma / self.delta, self.alpha / self.beta])


def lotka_volterra_invariant(x, y, alpha: float = 1.0, beta: float = 1.0, delta: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    r"""Conserved quantity of the Lotka-Volterra system.

    .. math:: V(x, y) = \delta x - \gamma \ln x + \beta y - \alpha \ln y

    Differentiating along a trajectory gives :math:`\dot V = (\delta -
    \gamma/x)\dot x + (\beta - \alpha/y)\dot y = 0`, so orbits are level
    sets of :math:`V` -- closed curves around the minimum at the
    coexistence fixed point (Volterra, 1926).

    Parameters
    ----------
    x, y : float or array-like
        Prey and predator populations (``> 0``).
    alpha, beta, delta, gamma : float

    Returns
    -------
    ndarray
        :math:`V(x, y)`.

    Examples
    --------
    >>> float(lotka_volterra_invariant(1.0, 1.0))  # minimum, at the fixed point
    2.0
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return delta * x - gamma * np.log(x) + beta * y - alpha * np.log(y)
