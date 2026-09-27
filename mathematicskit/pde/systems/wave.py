r"""The 1D wave equation :math:`u_{tt} = c^2 u_{xx}` on a string with fixed ends.

Semi-discretized in space, the wave equation becomes the separable
second-order system :math:`U'' = c^2 L U`, exactly the
``pos'' = force(pos, t)`` form that
:func:`mathematicskit.integrators.leapfrog_integrate` (velocity Verlet)
advances symplectically. That leapfrog scheme is stable iff the Courant
number :math:`c\,\Delta t/\Delta x \le 1` -- the CFL condition (Courant,
Friedrichs & Lewy, 1928) -- and at exactly :math:`\nu = 1` it reproduces
d'Alembert's solution at the grid points. RK4 on the first-order form
:math:`(U, V)' = (V, c^2 L U)` is also available; being non-symplectic, it
slowly damps the discrete energy.

:func:`dalembert_solution` is d'Alembert's (1747) traveling-wave
solution :math:`u = \tfrac12[f(x - ct) + f(x + ct)] + \tfrac{1}{2c}\int_{x-ct}^{x+ct} g`,
extended to a string with fixed ends by the method of images (odd
:math:`2L`-periodic extension). See Strauss, *Partial Differential
Equations: An Introduction*, 2nd ed., Ch. 2.1 and 4.1, and LeVeque,
*Finite Difference Methods for Ordinary and Partial Differential
Equations*, SIAM 2007, Ch. 10.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
from numba import njit

from mathematicskit.integrators import leapfrog_integrate
from mathematicskit.pde.core.base import MethodOfLinesPDE, PDESolution, _step_count
from mathematicskit.pde.systems.heat import InitialCondition, _sample
from mathematicskit.pde.systems.stability import RK4_IMAG_AXIS_LIMIT
from mathematicskit.pde.utils.operators import uniform_grid

__all__ = ["WaveEquation1D", "dalembert_solution"]


@njit(cache=True)
def _wave_force(pos, t, params):
    # params = [c^2 / dx^2]; pos = interior displacements, ends held at 0.
    k = params[0]
    n = pos.shape[0]
    out = np.empty(n)
    for i in range(n):
        left = pos[i - 1] if i > 0 else 0.0
        right = pos[i + 1] if i < n - 1 else 0.0
        out[i] = k * (left - 2.0 * pos[i] + right)
    return out


@njit(cache=True)
def _wave_rhs(state, t, params):
    m = state.shape[0] // 2
    out = np.empty(2 * m)
    out[:m] = state[m:]
    out[m:] = _wave_force(state[:m], t, params)
    return out


class WaveEquation1D(MethodOfLinesPDE):
    r"""A vibrating string :math:`u_{tt} = c^2 u_{xx}` on :math:`[0, L]` with :math:`u(0) = u(L) = 0`.

    Parameters
    ----------
    u0 : callable or ndarray, shape (n,)
        Initial displacement.
    v0 : callable or ndarray, shape (n,), optional
        Initial velocity (default zero).
    length : float
    n : int
        Grid points, both ends included.
    c : float
        Wave speed.

    Examples
    --------
    >>> import numpy as np
    >>> string = WaveEquation1D(lambda x: np.sin(np.pi * x), n=101)
    >>> sol = string.solve(1.0, dt=0.01, method="leapfrog")  # Courant number 1
    >>> sol.extra["stability"].stable
    True
    >>> bool(np.allclose(sol.final, -np.sin(np.pi * sol.x), atol=1e-3))  # half a period later
    True
    """

    def __init__(self, u0: InitialCondition, v0: Optional[InitialCondition] = None, length: float = 1.0, n: int = 101, c: float = 1.0):
        self.length, self.c = float(length), float(c)
        self.x, self.dx = uniform_grid(0.0, self.length, n)
        u_init = _sample(u0, self.x)
        v_init = np.zeros(n) if v0 is None else _sample(v0, self.x)
        self._m = n - 2
        self.state0 = np.concatenate([u_init[1:-1], v_init[1:-1]])
        self.params = np.array([self.c**2 / self.dx**2])
        self._rhs_njit = _wave_rhs

    def _to_field(self, states: np.ndarray) -> np.ndarray:
        zeros = np.zeros((states.shape[0], 1))
        return np.hstack([zeros, states[:, : self._m], zeros])

    def _integrate(self, t_final: float, dt: Optional[float], method: str, **kwargs):
        m = self._m
        if method == "leapfrog":
            if dt is None:
                raise ValueError("dt is required for method='leapfrog'")
            n_steps, dt = _step_count(t_final, dt)
            return leapfrog_integrate(_wave_force, self.state0[:m].copy(), self.state0[m:].copy(), 0.0, dt, n_steps, self.params)
        ts, states = super()._integrate(t_final, dt, method, **kwargs)
        return ts, states[:, :m], states[:, m:]

    def max_stable_dt(self, method: str = "leapfrog") -> float:
        r"""``dx / c`` for ``"leapfrog"`` (the CFL limit), :math:`\sqrt2\,dx/c` for ``"rk4"``, ``inf`` otherwise."""
        if method == "leapfrog":
            return self.dx / abs(self.c)
        if method == "rk4":
            return RK4_IMAG_AXIS_LIMIT * self.dx / (2.0 * abs(self.c))
        return np.inf

    def solve(self, t_final: float, dt: Optional[float] = None, method: str = "leapfrog", save_every: int = 1, **kwargs) -> PDESolution:
        """Integrate from ``t = 0`` to `t_final` with ``"leapfrog"`` (default), ``"rk4"``, or ``"dopri5"``.

        See :meth:`MethodOfLinesPDE.solve`. The velocity field is returned
        in ``extra["v"]`` (same shape as ``u``).
        """
        return super().solve(t_final, dt=dt, method=method, save_every=save_every, **kwargs)

    def energy(self, solution: PDESolution) -> np.ndarray:
        r"""Discrete energy :math:`E = \tfrac12 \sum_i v_i^2\,\Delta x + \tfrac12 c^2 \sum_i \big((u_{i+1} - u_i)/\Delta x\big)^2 \Delta x` at each saved time.

        Conserved by the exact PDE; leapfrog keeps it bounded (oscillating
        near its initial value), RK4 slowly dissipates it.

        Parameters
        ----------
        solution : PDESolution
            From :meth:`solve`.

        Returns
        -------
        ndarray, shape (n_t,)
        """
        u, v = solution.u, solution.extra["v"]
        kinetic = 0.5 * np.sum(v**2, axis=1) * self.dx
        strain = np.diff(u, axis=1) / self.dx
        potential = 0.5 * self.c**2 * np.sum(strain**2, axis=1) * self.dx
        return kinetic + potential


def _odd_periodic(f: Callable, length: float) -> Callable:
    def extended(s):
        s = np.mod(s, 2.0 * length)
        return np.where(s <= length, f(np.minimum(s, length)), -f(np.clip(2.0 * length - s, 0.0, length)))

    return extended


def _even_periodic(f: Callable, length: float) -> Callable:
    def extended(s):
        s = np.mod(s, 2.0 * length)
        return np.where(s <= length, f(np.minimum(s, length)), f(np.clip(2.0 * length - s, 0.0, length)))

    return extended


def dalembert_solution(f: Callable, x, t: float, c: float = 1.0, g_antiderivative: Optional[Callable] = None, length: Optional[float] = None) -> np.ndarray:
    r"""D'Alembert's solution of the wave equation (d'Alembert, 1747).

    .. math:: u(x, t) = \frac{f(x - ct) + f(x + ct)}{2} + \frac{G(x + ct) - G(x - ct)}{2c}

    with :math:`f` the initial displacement and :math:`G' = g` the initial
    velocity. On the whole line this is exact as written. Passing `length`
    solves the fixed-end string on :math:`[0, L]` instead, by replacing
    :math:`f` (and :math:`g`) with their odd :math:`2L`-periodic
    extensions (method of images; Strauss, Ch. 4.1).

    Parameters
    ----------
    f : callable
        Initial displacement, vectorized over NumPy arrays.
    x : array-like
    t : float
    c : float
    g_antiderivative : callable, optional
        An antiderivative :math:`G` of the initial velocity (on
        :math:`[0, L]` when `length` is given); zero velocity if omitted.
    length : float, optional
        String length for the fixed-end problem.

    Returns
    -------
    ndarray, same shape as `x`

    Examples
    --------
    >>> import numpy as np
    >>> bump = lambda s: np.exp(-100 * s**2)
    >>> u = dalembert_solution(bump, np.array([-1.0, 0.0, 1.0]), t=1.0)
    >>> np.round(u, 6).tolist()  # the bump splits into two half-height copies
    [0.5, 0.0, 0.5]
    """
    x = np.asarray(x, dtype=np.float64)
    F = f if length is None else _odd_periodic(f, length)
    u = 0.5 * (F(x - c * t) + F(x + c * t))
    if g_antiderivative is not None:
        G = g_antiderivative if length is None else _even_periodic(g_antiderivative, length)
        u = u + (G(x + c * t) - G(x - c * t)) / (2.0 * c)
    return np.asarray(u, dtype=np.float64)
