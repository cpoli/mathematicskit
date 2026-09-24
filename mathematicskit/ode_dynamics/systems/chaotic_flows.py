r"""The Rossler system: a minimal three-dimensional chaotic flow.

See Rossler (1976) and Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed.,
Ch. 12.3.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from mathematicskit.ode_dynamics.core.base import FlowSystem

__all__ = ["RosslerSystem", "rossler_fixed_points"]


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
