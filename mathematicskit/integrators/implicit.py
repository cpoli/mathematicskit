"""Implicit integration for stiff systems: backward Euler, Radau, and BDF.

A system is *stiff* when it mixes time scales so different that an
explicit method's step size is capped by the fastest, already-decayed
mode rather than by the accuracy needed for the slow dynamics. For the
test equation ``y' = lam * y`` an explicit Runge-Kutta step multiplies
``y`` by a polynomial ``R(z)`` in ``z = lam * dt``, and ``|R(z)| <= 1``
only on a bounded region (RK4's real-axis limit is ``z ~ -2.785``), so
``dt`` must shrink like ``1 / |lam|`` however smooth the solution is.
Backward Euler's ``R(z) = 1 / (1 - z)`` has ``|R(z)| < 1`` on the whole
left half-plane (A-stability; Dahlquist 1963), so ``dt`` is set by
accuracy alone -- at the price of solving a nonlinear system per step.

:func:`implicit_euler_step` / :func:`implicit_euler_integrate` hand-roll
backward Euler (Newton's method on ``y_new - y - dt * f(y_new) = 0`` with a
forward-difference Jacobian) to make that trade-off visible, sharing the
``rhs(state, t, params)`` convention of
:mod:`mathematicskit.integrators.fixed_step`. For production use,
:func:`stiff_integrate` wraps :func:`scipy.integrate.solve_ivp`'s
adaptive, high-order implicit methods: Radau IIA (order 5; Hairer &
Wanner, *Solving ODEs II*, sec. IV.8) and the variable-order BDF /
numerical differentiation formulas (Curtiss & Hirschfelder 1952; Gear
1971; Shampine & Reichelt 1997).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from mathematicskit.integrators.fixed_step import RHSFunc

__all__ = ["implicit_euler_step", "implicit_euler_integrate", "stiff_integrate"]

_SQRT_EPS = 1.4901161193847656e-08  # sqrt(float64 machine epsilon)


@njit(cache=True)
def _fd_jacobian(
    rhs: RHSFunc,
    state: NDArray[np.float64],
    t: float,
    params: NDArray[np.float64],
    f0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Forward-difference Jacobian ``d rhs / d state`` at ``state``."""
    dim = state.shape[0]
    jac = np.empty((dim, dim))
    for j in range(dim):
        h = _SQRT_EPS * max(1.0, abs(state[j]))
        shifted = state.copy()
        shifted[j] += h
        jac[:, j] = (rhs(shifted, t, params) - f0) / h
    return jac


@njit(cache=True)
def implicit_euler_step(
    rhs: RHSFunc,
    state: NDArray[np.float64],
    t: float,
    dt: float,
    params: NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """Single backward (implicit) Euler step, solved by Newton's method.

    Solves ``G(y) = y - state - dt * rhs(y, t + dt, params) = 0`` with
    Newton iterations ``(I - dt * J) delta = -G(y)``, where ``J`` is a
    forward-difference Jacobian of `rhs` re-evaluated every iteration.

    Parameters
    ----------
    rhs : callable
        Numba-jitted right-hand-side function ``rhs(state, t, params) ->
        ndarray``.
    state : ndarray of float, shape (dim,)
        Current state vector.
    t : float
        Current time.
    dt : float
        Step size.
    params : ndarray of float
        Parameter vector passed through to `rhs`.
    tol : float, default=1e-10
        Newton stops once ``max|delta| <= tol * (1 + max|y|)``.
    max_iter : int, default=50
        Maximum Newton iterations.

    Returns
    -------
    ndarray of float, shape (dim,)
        The state at ``t + dt``.

    Raises
    ------
    RuntimeError
        If Newton's method does not converge within `max_iter` iterations.
    """
    t_new = t + dt
    y = state.copy()
    eye = np.eye(state.shape[0])
    for _ in range(max_iter):
        f = rhs(y, t_new, params)
        residual = y - state - dt * f
        delta = np.linalg.solve(eye - dt * _fd_jacobian(rhs, y, t_new, params, f), -residual)
        y = y + delta
        if np.max(np.abs(delta)) <= tol * (1.0 + np.max(np.abs(y))):
            return y
    raise RuntimeError("implicit_euler_step: Newton iteration did not converge")


@njit(cache=True)
def implicit_euler_integrate(
    rhs: RHSFunc,
    state0: NDArray[np.float64],
    t0: float,
    dt: float,
    n_steps: int,
    params: NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate ``n_steps`` of backward Euler starting from ``state0``.

    First-order accurate but A-stable: for any ``dt > 0`` a decaying mode
    stays decaying, so stiff systems can be stepped at a `dt` far beyond
    the explicit stability limit (compare
    :func:`~mathematicskit.integrators.fixed_step.rk4_integrate`).

    Parameters
    ----------
    rhs : callable
        Numba-jitted right-hand-side function ``rhs(state, t, params) ->
        ndarray``.
    state0 : ndarray of float, shape (dim,)
        Initial state vector.
    t0 : float
        Initial time.
    dt : float
        Step size.
    n_steps : int
        Number of integration steps.
    params : ndarray of float
        Parameter vector passed through to `rhs`.
    tol, max_iter
        Newton-iteration controls, see :func:`implicit_euler_step`.

    Returns
    -------
    times : ndarray of float, shape (n_steps + 1,)
        Time at each step, starting at `t0`.
    states : ndarray of float, shape (n_steps + 1, dim)
        State at each step, starting at `state0`.
    """
    dim = state0.shape[0]
    states = np.empty((n_steps + 1, dim))
    times = np.empty(n_steps + 1)
    states[0] = state0
    times[0] = t0
    state = state0.copy()
    t = t0
    for i in range(n_steps):
        state = implicit_euler_step(rhs, state, t, dt, params, tol, max_iter)
        t = t0 + (i + 1) * dt
        states[i + 1] = state
        times[i + 1] = t
    return times, states


_STIFF_METHODS = ("Radau", "BDF", "LSODA")


def stiff_integrate(
    rhs: RHSFunc,
    state0: ArrayLike,
    t0: float,
    t_end: float,
    params: NDArray[np.float64],
    method: str = "Radau",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    t_eval: ArrayLike | None = None,
    jac: Callable[[NDArray[np.float64], float, NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate a stiff system with :func:`scipy.integrate.solve_ivp`.

    Adapts the ``rhs(state, t, params)`` convention to scipy's
    ``fun(t, y)`` and returns the same ``(times, states)`` layout as
    :func:`~mathematicskit.integrators.adaptive.dopri5_integrate`, so the
    two are drop-in comparable.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(state, t, params) -> ndarray``; an ``@njit``
        function or a plain Python one.
    state0 : array_like of float, shape (dim,)
        Initial state vector.
    t0, t_end : float
        Integration interval.
    params : ndarray of float
        Parameter vector passed through to `rhs` (and `jac`).
    method : {"Radau", "BDF", "LSODA"}, default="Radau"
        Implicit Radau IIA (order 5), variable-order BDF (orders 1-5), or
        LSODA (switches automatically between Adams and BDF).
    rtol, atol : float
        Relative and absolute error tolerances.
    t_eval : array_like of float, optional
        Times at which to report the solution; by default the solver's own
        accepted steps are returned.
    jac : callable, optional
        Analytic Jacobian ``jac(state, t, params) -> ndarray (dim, dim)``;
        estimated by finite differences if omitted.

    Returns
    -------
    times : ndarray of float, shape (n,)
        Output times, from `t0` to `t_end`.
    states : ndarray of float, shape (n, dim)
        State at each output time.

    Raises
    ------
    ValueError
        If `method` is not one of the supported implicit methods.
    RuntimeError
        If the solver fails.

    Examples
    --------
    >>> import numpy as np
    >>> def decay(state, t, params):
    ...     return -params[0] * state
    >>> ts, ys = stiff_integrate(decay, [1.0], 0.0, 1.0, np.array([1e4]), method="BDF")
    >>> len(ts) < 200, bool(abs(ys[-1, 0]) < 1e-8)
    (True, True)
    """
    if method not in _STIFF_METHODS:
        raise ValueError(f"method must be one of {_STIFF_METHODS}, got {method!r}")
    sol = solve_ivp(
        lambda t, y: rhs(y, t, params),
        (t0, t_end),
        np.asarray(state0, dtype=np.float64),
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        jac=None if jac is None else (lambda t, y: jac(y, t, params)),
    )
    if not sol.success:
        raise RuntimeError(f"stiff_integrate ({method}) failed: {sol.message}")
    return sol.t, sol.y.T
