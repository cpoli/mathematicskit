r"""The FitzHugh-Nagumo model of an excitable neuron.

See FitzHugh (1961), Nagumo, Arimoto & Yoshizawa (1962), and Izhikevich,
*Dynamical Systems in Neuroscience* (MIT Press, 2007), Ch. 4.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from mathematicskit.ode_dynamics.core.base import FlowSystem

__all__ = ["FitzHughNagumo", "fitzhugh_nagumo_fixed_point", "fitzhugh_nagumo_hopf_currents"]


@lru_cache(maxsize=64)
def _make_fhn_rhs(a: float, b: float, eps: float, current: float):
    @njit(cache=True)
    def rhs(state, t, params):
        v, w = state[0], state[1]
        out = np.empty(2)
        out[0] = v - v * v * v / 3.0 - w + current
        out[1] = eps * (v + a - b * w)
        return out

    return rhs


class FitzHughNagumo(FlowSystem):
    r"""FitzHugh-Nagumo neuron model.

    :math:`\dot v = v - v^3/3 - w + I`, :math:`\dot w = \varepsilon(v +
    a - bw)`: a fast "membrane voltage" :math:`v` with a cubic nullcline
    and a slow recovery variable :math:`w`. For small injected current
    :math:`I` the rest state is stable but *excitable* -- a large enough
    kick fires a single large spike before returning to rest; for
    :math:`I` between the two values from
    :func:`fitzhugh_nagumo_hopf_currents`, the rest state is unstable and
    the neuron fires periodically (a relaxation limit cycle).

    Parameters
    ----------
    state0 : array-like, shape (2,)
        Initial ``(v, w)``.
    a, b, eps : float
        Model parameters (FitzHugh's classic values ``0.7, 0.8, 0.08``).
    current : float
        Injected current :math:`I`.

    Examples
    --------
    >>> system = FitzHughNagumo([-1.2, -0.6], current=0.0)
    >>> result = system.integrate((0.0, 200.0), dt=1e-2, method="rk4")
    >>> v_star, w_star = fitzhugh_nagumo_fixed_point(current=0.0)
    >>> bool(abs(result.y[-1, 0] - v_star) < 1e-3)
    True
    """

    def __init__(self, state0, a: float = 0.7, b: float = 0.8, eps: float = 0.08, current: float = 0.0):
        self.a, self.b, self.eps, self.current = float(a), float(b), float(eps), float(current)
        self._rhs_njit = _make_fhn_rhs(self.a, self.b, self.eps, self.current)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        v, w = state[0], state[1]
        return np.array([v - v**3 / 3.0 - w + self.current, self.eps * (v + self.a - self.b * w)])


def fitzhugh_nagumo_fixed_point(current: float = 0.0, a: float = 0.7, b: float = 0.8) -> np.ndarray:
    r"""The (unique, for :math:`0 < b < 1`) fixed point of FitzHugh-Nagumo.

    Intersecting the nullclines :math:`w = v - v^3/3 + I` and :math:`w =
    (v + a)/b` gives the cubic :math:`-v^3/3 + (1 - 1/b)v + I - a/b = 0`,
    which is strictly decreasing (so has one real root) when :math:`b <
    1`.

    Parameters
    ----------
    current : float
    a, b : float

    Returns
    -------
    ndarray, shape (2,)
        ``(v*, w*)``.

    Examples
    --------
    >>> np.round(fitzhugh_nagumo_fixed_point(0.0), 4)
    array([-1.1994, -0.6243])
    """
    if not 0.0 < b < 1.0:
        raise ValueError("b must lie in (0, 1) for a unique fixed point")
    roots = np.roots([-1.0 / 3.0, 0.0, 1.0 - 1.0 / b, current - a / b])
    v = float(roots[np.argmin(np.abs(roots.imag))].real)
    return np.array([v, (v + a) / b])


def fitzhugh_nagumo_hopf_currents(a: float = 0.7, b: float = 0.8, eps: float = 0.08) -> np.ndarray:
    r"""Injected currents at which the rest state loses/regains stability.

    The Jacobian at the fixed point is :math:`\begin{pmatrix} 1 - v^2 &
    -1 \\ \varepsilon & -\varepsilon b\end{pmatrix}`, whose determinant
    :math:`\varepsilon(1 - b + bv^2)` is positive for :math:`b<1`, so
    stability changes exactly when the trace :math:`1 - v^2 - \varepsilon
    b` vanishes: :math:`v_H = \pm\sqrt{1 - \varepsilon b}`. Substituting
    into the nullcline relation gives the Hopf currents

    .. math:: I_H = \frac{v_H + a}{b} - v_H + \frac{v_H^3}{3}.

    Parameters
    ----------
    a, b, eps : float

    Returns
    -------
    ndarray, shape (2,)
        ``[I_low, I_high]``; the rest state is unstable for ``I_low < I <
        I_high``.

    Examples
    --------
    >>> np.round(fitzhugh_nagumo_hopf_currents(), 4)
    array([0.3313, 1.4187])
    """
    if eps * b >= 1.0:
        raise ValueError("no Hopf bifurcation when eps * b >= 1")
    v = np.array([-1.0, 1.0]) * np.sqrt(1.0 - eps * b)
    return (v + a) / b - v + v**3 / 3.0
