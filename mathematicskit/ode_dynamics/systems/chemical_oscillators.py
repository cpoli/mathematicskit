r"""The Brusselator: a model chemical oscillator with a closed-form Hopf
bifurcation.

See Prigogine & Lefever (1968) and Strogatz, *Nonlinear Dynamics and
Chaos*, 2nd ed., Exercise 8.3.1.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from mathematicskit.ode_dynamics.core.base import FlowSystem

__all__ = ["Brusselator", "brusselator_hopf_threshold"]


@lru_cache(maxsize=64)
def _make_brusselator_rhs(a: float, b: float):
    @njit(cache=True)
    def rhs(state, t, params):
        x, y = state[0], state[1]
        out = np.empty(2)
        out[0] = a - (b + 1.0) * x + x * x * y
        out[1] = b * x - x * x * y
        return out

    return rhs


class Brusselator(FlowSystem):
    r"""The Brusselator reaction scheme.

    :math:`\dot x = a - (b+1)x + x^2y`, :math:`\dot y = bx - x^2y`, the
    rate equations of the hypothetical reactions :math:`A \to X`,
    :math:`B + X \to Y + D`, :math:`2X + Y \to 3X`, :math:`X \to E` with
    the concentrations of :math:`A, B` held fixed. The unique fixed point
    :math:`(a,\ b/a)` loses stability in a Hopf bifurcation at :math:`b
    = 1 + a^2` (:func:`brusselator_hopf_threshold`), beyond which the
    concentrations oscillate on a stable limit cycle.

    Parameters
    ----------
    state0 : array-like, shape (2,)
        Initial concentrations ``(x, y)``.
    a, b : float
        Positive feed concentrations.

    Examples
    --------
    >>> system = Brusselator([1.0, 1.0], a=1.0, b=1.5)  # b < 1 + a^2 = 2
    >>> result = system.integrate((0.0, 100.0), dt=1e-2, method="rk4")
    >>> np.round(result.y[-1], 4)
    array([1. , 1.5])
    """

    def __init__(self, state0, a: float = 1.0, b: float = 3.0):
        self.a, self.b = float(a), float(b)
        self._rhs_njit = _make_brusselator_rhs(self.a, self.b)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        x, y = state[0], state[1]
        return np.array([self.a - (self.b + 1.0) * x + x * x * y, self.b * x - x * x * y])

    def fixed_point(self) -> np.ndarray:
        """The unique fixed point ``(a, b/a)``."""
        return np.array([self.a, self.b / self.a])


def brusselator_hopf_threshold(a: float) -> float:
    r"""Critical :math:`b` for the Brusselator's Hopf bifurcation.

    At :math:`(a, b/a)` the Jacobian is :math:`\begin{pmatrix} b - 1 &
    a^2 \\ -b & -a^2\end{pmatrix}`, with determinant :math:`a^2 > 0` and
    trace :math:`b - 1 - a^2`. The fixed point is therefore a stable
    focus/node for :math:`b < 1 + a^2` and unstable beyond it, where a
    limit cycle appears.

    Parameters
    ----------
    a : float

    Returns
    -------
    float
        :math:`b_c = 1 + a^2`.

    Examples
    --------
    >>> brusselator_hopf_threshold(1.0)
    2.0
    """
    return float(1.0 + a * a)
