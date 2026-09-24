r"""The Van der Pol oscillator: a canonical example of a stable limit cycle.

See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 7.4 ("Relaxation
Oscillations"), and the original Van der Pol (1926) paper on vacuum-tube
circuits. :func:`bendixson_criterion` implements Bendixson's (1901)
negative criterion for ruling limit cycles *out*.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from numba import njit

from mathematicskit.ode_dynamics.core.base import BendixsonResult, FlowSystem

__all__ = ["VanDerPolOscillator", "estimate_limit_cycle_amplitude", "bendixson_criterion"]


@lru_cache(maxsize=64)
def _make_vdp_rhs(mu: float):
    @njit(cache=True)
    def rhs(state, t, params):
        x, y = state[0], state[1]
        out = np.empty(2)
        out[0] = y
        out[1] = mu * (1.0 - x * x) * y - x
        return out

    return rhs


class VanDerPolOscillator(FlowSystem):
    r"""Van der Pol oscillator :math:`\ddot x - \mu(1-x^2)\dot x + x = 0`.

    Written as a first-order system :math:`\dot x = y`, :math:`\dot y =
    \mu(1-x^2)y - x`. For any :math:`\mu > 0`, every trajectory except
    the (unstable) origin converges to a *unique* stable limit cycle
    (a consequence of the Poincare-Bendixson theorem, since the system
    is dissipative for :math:`|x|>1` and anti-dissipative for
    :math:`|x|<1`) -- unlike a linear center, the cycle's amplitude and
    shape are independent of initial conditions. See Strogatz,
    *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 7.4.

    Parameters
    ----------
    state0 : array-like, shape (2,)
        Initial ``(x, y)``.
    mu : float
        Nonlinear damping strength (``mu=0`` reduces to the harmonic
        oscillator, a center with no limit cycle).

    Examples
    --------
    >>> system = VanDerPolOscillator([2.0, 0.0], mu=1.0)
    >>> result = system.integrate((0.0, 50.0), dt=1e-3, method="rk4")
    >>> result.y.shape[1]
    2
    """

    def __init__(self, state0, mu: float = 1.0):
        self.mu = float(mu)
        self._rhs_njit = _make_vdp_rhs(self.mu)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        x, y = state[0], state[1]
        return np.array([y, self.mu * (1.0 - x * x) * y - x])


def estimate_limit_cycle_amplitude(mu: float, t_transient: float = 200.0, t_observe: float = 50.0, dt: float = 1e-3) -> float:
    """Estimate the Van der Pol limit cycle's amplitude (``max |x|``) after
    transients have decayed.

    Parameters
    ----------
    mu : float
    t_transient : float
        Integration time discarded to let the trajectory approach the cycle.
    t_observe : float
        Additional integration time over which the amplitude is measured.
    dt : float

    Returns
    -------
    float
        ``max(|x|)`` over the observation window.

    Examples
    --------
    >>> amp = estimate_limit_cycle_amplitude(mu=1.0, t_transient=100.0, t_observe=30.0)
    >>> 1.8 < amp < 2.2
    True
    """
    system = VanDerPolOscillator([0.5, 0.0], mu=mu)
    system.integrate((0.0, t_transient), dt=dt, method="rk4")
    result = system.integrate((system.t, system.t + t_observe), dt=dt, method="rk4")
    return float(np.max(np.abs(result.y[:, 0])))


def bendixson_criterion(f: Callable[[float, float], tuple], x_range, y_range, n: int = 101, h: float = 1e-6) -> BendixsonResult:
    r"""Bendixson's negative criterion on a rectangle.

    For a planar system :math:`\dot x = f(x, y)`, :math:`\dot y = g(x,
    y)`, if the divergence :math:`\partial f/\partial x + \partial
    g/\partial y` is not identically zero and does not change sign on a
    simply connected region :math:`D`, then no periodic orbit lies
    entirely in :math:`D` (Bendixson, 1901). Proof sketch: by Green's
    theorem, the divergence integrated over the region enclosed by a
    closed orbit equals the flux of the field across the orbit, which is
    zero because the field is tangent to it. This function samples the
    divergence (by central differences) on an ``n x n`` grid and reports
    whether it is strictly of one sign there -- a numerical check, not a
    proof.

    Parameters
    ----------
    f : callable
        ``f(x, y) -> (dx, dy)``, accepting NumPy arrays elementwise.
    x_range, y_range : tuple of float
        ``(min, max)`` of the rectangle.
    n : int
        Grid resolution per axis.
    h : float
        Central-difference step.

    Returns
    -------
    BendixsonResult

    Examples
    --------
    >>> damped = lambda x, y: (y, -x - 0.5 * y)  # divergence = -0.5
    >>> bendixson_criterion(damped, (-2, 2), (-2, 2)).rules_out_periodic_orbits
    True
    >>> vdp = lambda x, y: (y, (1 - x**2) * y - x)  # divergence = 1 - x^2
    >>> bendixson_criterion(vdp, (-3, 3), (-3, 3)).rules_out_periodic_orbits
    False
    """
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(xs, ys)
    dfdx = (np.asarray(f(X + h, Y)[0]) - np.asarray(f(X - h, Y)[0])) / (2.0 * h)
    dgdy = (np.asarray(f(X, Y + h)[1]) - np.asarray(f(X, Y - h)[1])) / (2.0 * h)
    div = np.broadcast_to(dfdx + dgdy, X.shape).astype(np.float64)
    tol = 1e-8
    one_sign = bool(np.all(div > tol) or np.all(div < -tol))
    return BendixsonResult(X=X, Y=Y, divergence=div, rules_out_periodic_orbits=one_sign)
