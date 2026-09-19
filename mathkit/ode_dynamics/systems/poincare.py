r"""Poincare sections for higher-dimensional (here: periodically-driven,
effectively 3D via the phase of the drive) flows, demonstrated on the
forced Duffing oscillator.

See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 9 ("Poincare
Maps") and Ch. 12 ("Chaos and the Duffing oscillator" case study).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from mathkit.ode_dynamics.core.base import FlowSystem

__all__ = ["DuffingOscillator", "stroboscopic_poincare_section"]


@lru_cache(maxsize=64)
def _make_duffing_rhs(delta: float, alpha: float, beta: float, gamma: float, omega: float):
    @njit(cache=True)
    def rhs(state, t, params):
        x, y = state[0], state[1]
        out = np.empty(2)
        out[0] = y
        out[1] = gamma * np.cos(omega * t) - delta * y - alpha * x - beta * x**3
        return out

    return rhs


class DuffingOscillator(FlowSystem):
    r"""Periodically-driven Duffing oscillator.

    :math:`\ddot x + \delta \dot x + \alpha x + \beta x^3 = \gamma
    \cos(\omega t)`. Explicit time-dependence (through the drive) makes
    the ``(x, y)`` phase portrait alone insufficient to see the full
    (3D, including drive phase) state space; a *stroboscopic* Poincare
    section -- sampling ``(x, y)`` once per drive period -- reduces the
    long-time behavior back to a 2D picture, revealing periodic points
    for periodic motion and a fractal attractor for chaotic parameters.
    See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 12.3.

    Parameters
    ----------
    state0 : array-like, shape (2,)
    delta : float
        Damping.
    alpha, beta : float
        Linear and cubic stiffness (``alpha<0, beta>0`` gives the classic
        double-well potential).
    gamma, omega : float
        Drive amplitude and angular frequency.

    Examples
    --------
    >>> system = DuffingOscillator([1.0, 0.0], delta=0.3, alpha=-1.0, beta=1.0, gamma=0.5, omega=1.2)
    >>> result = system.integrate((0.0, 10.0), dt=1e-3, method="rk4")
    >>> result.y.shape[1]
    2
    """

    def __init__(self, state0, delta: float = 0.3, alpha: float = -1.0, beta: float = 1.0, gamma: float = 0.3, omega: float = 1.2):
        self.delta, self.alpha, self.beta, self.gamma, self.omega = float(delta), float(alpha), float(beta), float(gamma), float(omega)
        self._rhs_njit = _make_duffing_rhs(self.delta, self.alpha, self.beta, self.gamma, self.omega)
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        x, y = state[0], state[1]
        return np.array([y, self.gamma * np.cos(self.omega * t) - self.delta * y - self.alpha * x - self.beta * x**3])


def stroboscopic_poincare_section(system: DuffingOscillator, n_periods: int, dt: float = 1e-3, n_transient_periods: int = 20):
    """Sample ``(x, y)`` once per drive period after discarding transients.

    Parameters
    ----------
    system : DuffingOscillator
    n_periods : int
        Number of drive periods to record after the transient.
    dt : float
        Fixed integration step (must divide the drive period reasonably
        finely).
    n_transient_periods : int
        Number of drive periods discarded before recording begins.

    Returns
    -------
    xs, ys : ndarray, shape (n_periods,)
        The stroboscopic (once-per-period) samples.

    Examples
    --------
    >>> system = DuffingOscillator([1.0, 0.0], delta=0.3, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2)
    >>> xs, ys = stroboscopic_poincare_section(system, n_periods=5, n_transient_periods=5, dt=1e-2)
    >>> len(xs)
    5
    """
    period = 2.0 * np.pi / system.omega
    n_steps_per_period = max(int(round(period / dt)), 1)
    actual_dt = period / n_steps_per_period

    system.integrate((0.0, n_transient_periods * period), dt=actual_dt, method="rk4")

    xs = np.empty(n_periods)
    ys = np.empty(n_periods)
    for k in range(n_periods):
        result = system.integrate((system.t, system.t + period), dt=actual_dt, method="rk4")
        xs[k] = result.y[-1, 0]
        ys[k] = result.y[-1, 1]
    return xs, ys
