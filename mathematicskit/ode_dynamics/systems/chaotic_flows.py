r"""Three-dimensional chaotic flows: the Lorenz and Rossler systems.

See Lorenz (1963), Rossler (1976), and Strogatz, *Nonlinear Dynamics and
Chaos*, 2nd ed., Ch. 9 and 12.3.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from mathematicskit.ode_dynamics.core.base import FlowSystem

__all__ = ["LorenzSystem", "lorenz_fixed_points", "RosslerSystem", "rossler_fixed_points"]


@lru_cache(maxsize=64)
def _make_lorenz_rhs(sigma: float, rho: float, beta: float):
    @njit(cache=True)
    def rhs(state, t, params):
        x, y, z = state[0], state[1], state[2]
        out = np.empty(3)
        out[0] = sigma * (y - x)
        out[1] = x * (rho - z) - y
        out[2] = x * y - beta * z
        return out

    return rhs


class LorenzSystem(FlowSystem):
    r"""Lorenz's convection model (Lorenz, 1963).

    :math:`\dot x = \sigma(y - x)`, :math:`\dot y = x(\rho - z) - y`,
    :math:`\dot z = xy - \beta z`. For Lorenz's classic values
    :math:`\sigma = 10`, :math:`\rho = 28`, :math:`\beta = 8/3` every
    fixed point is unstable and trajectories settle onto the butterfly-
    shaped strange attractor, where nearby orbits separate exponentially
    (sensitive dependence on initial conditions).

    Parameters
    ----------
    state0 : array-like, shape (3,)
    sigma, rho, beta : float

    Examples
    --------
    >>> system = LorenzSystem([1.0, 1.0, 1.0])
    >>> result = system.integrate((0.0, 10.0), dt=1e-2, method="rk4")
    >>> result.y.shape
    (1001, 3)
    """

    def __init__(self, state0, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        self.sigma, self.rho, self.beta = float(sigma), float(rho), float(beta)
        self._rhs_njit = _make_lorenz_rhs(self.sigma, self.rho, self.beta)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        x, y, z = state[0], state[1], state[2]
        return np.array([self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z])


def lorenz_fixed_points(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    r"""Closed-form fixed points of the Lorenz system.

    The origin is always a fixed point; for :math:`\rho > 1` the two
    convection-roll states

    .. math:: C_\pm = \left(\pm\sqrt{\beta(\rho - 1)},\
              \pm\sqrt{\beta(\rho - 1)},\ \rho - 1\right)

    also exist (Lorenz, 1963; Strogatz, Sec. 9.2). ``sigma`` does not
    affect their location, only their stability.

    Parameters
    ----------
    sigma, rho, beta : float

    Returns
    -------
    ndarray, shape (1, 3) or (3, 3)
        The origin first, then :math:`C_+` and :math:`C_-` when
        :math:`\rho > 1`.

    Examples
    --------
    >>> fps = lorenz_fixed_points()
    >>> fps.shape
    (3, 3)
    >>> [round(float(v), 4) for v in fps[1]]
    [8.4853, 8.4853, 27.0]
    >>> lorenz_fixed_points(rho=0.5).shape
    (1, 3)
    """
    origin = np.zeros((1, 3))
    if rho <= 1.0:
        return origin
    r = np.sqrt(beta * (rho - 1.0))
    return np.vstack([origin, [r, r, rho - 1.0], [-r, -r, rho - 1.0]])


@lru_cache(maxsize=64)
def _make_rossler_rhs(a: float, b: float, c: float):
    @njit(cache=True)
    def rhs(state, t, params):
        x, y, z = state[0], state[1], state[2]
        out = np.empty(3)
        out[0] = -y - z
        out[1] = x + a * y
        out[2] = b + z * (x - c)
        return out

    return rhs


class RosslerSystem(FlowSystem):
    r"""Rossler's chaotic flow.

    :math:`\dot x = -y - z`, :math:`\dot y = x + ay`, :math:`\dot z = b +
    z(x - c)`. Only one term (:math:`zx`) is nonlinear. Trajectories
    spiral outward in the :math:`(x, y)` plane until :math:`x` exceeds
    :math:`c`, when :math:`z` spikes and folds them back toward the
    center -- a stretch-and-fold mechanism producing a chaotic attractor
    for the classic values :math:`a = b = 0.2`, :math:`c = 5.7`.

    Parameters
    ----------
    state0 : array-like, shape (3,)
    a, b, c : float

    Examples
    --------
    >>> system = RosslerSystem([1.0, 1.0, 0.0])
    >>> result = system.integrate((0.0, 100.0), dt=1e-2, method="rk4")
    >>> result.y.shape
    (10001, 3)
    """

    def __init__(self, state0, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        self.a, self.b, self.c = float(a), float(b), float(c)
        self._rhs_njit = _make_rossler_rhs(self.a, self.b, self.c)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        x, y, z = state[0], state[1], state[2]
        return np.array([-y - z, x + self.a * y, self.b + z * (x - self.c)])


def rossler_fixed_points(a: float = 0.2, b: float = 0.2, c: float = 5.7) -> np.ndarray:
    r"""Closed-form fixed points of the Rossler system.

    Setting the right-hand side to zero gives :math:`z = -y`, :math:`x =
    -ay`, and then :math:`ay^2 + cy + b = 0`, so

    .. math:: y_\pm = \frac{-c \pm \sqrt{c^2 - 4ab}}{2a}, \qquad
              (x, y, z) = (-a y_\pm,\ y_\pm,\ -y_\pm),

    which are real when :math:`c^2 \ge 4ab`.

    Parameters
    ----------
    a, b, c : float

    Returns
    -------
    ndarray, shape (2, 3)
        One fixed point per row (the one near the origin first).

    Examples
    --------
    >>> fps = rossler_fixed_points()
    >>> [round(float(v), 4) for v in fps[0]]
    [0.007, -0.0351, 0.0351]
    >>> [round(float(v), 4) for v in fps[1]]
    [5.693, -28.4649, 28.4649]
    """
    disc = c * c - 4.0 * a * b
    if disc < 0:
        raise ValueError("no real fixed points when c**2 < 4ab")
    ys = np.array([(-c + np.sqrt(disc)) / (2.0 * a), (-c - np.sqrt(disc)) / (2.0 * a)])
    return np.column_stack([-a * ys, ys, -ys])
