r"""Explicit linear multistep integration: the Adams-Bashforth methods.

A one-step method such as RK4 spends several right-hand-side evaluations
per step and then forgets them. Bashforth and Adams (1883) reused them
instead: integrate the polynomial that interpolates the last :math:`k`
slopes :math:`f_n, f_{n-1}, \ldots, f_{n-k+1}` over the next step,

.. math::

   y_{n+1} = y_n + h \sum_{j=0}^{k-1} \beta_j\, f_{n-j},

which is order :math:`k` at the cost of *one* new evaluation per step.
The price is a smaller stability region than a one-step method of the
same order (see :mod:`mathematicskit.integrators.stability`), and the
need for :math:`k - 1` starting values, which :func:`adams_bashforth_integrate`
supplies with RK4 steps (Hairer, Nørsett & Wanner, *Solving ODEs I*,
2nd ed., sec. III.1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mathematicskit._jit import njit
from mathematicskit.integrators.fixed_step import RHSFunc, rk4_step

__all__ = ["ADAMS_BASHFORTH_COEFFICIENTS", "adams_bashforth_integrate"]

#: Row ``k - 1`` holds the order-``k`` Adams-Bashforth weights
#: :math:`\beta_0, \ldots, \beta_{k-1}` applied to :math:`f_n, f_{n-1},
#: \ldots` (zero-padded to length 4). Hairer, Nørsett & Wanner, Table III.1.1.
ADAMS_BASHFORTH_COEFFICIENTS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [3.0 / 2.0, -1.0 / 2.0, 0.0, 0.0],
        [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0, 0.0],
        [55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
    ]
)


@njit
def adams_bashforth_integrate(
    rhs: RHSFunc,
    state0: NDArray[np.float64],
    t0: float,
    dt: float,
    n_steps: int,
    params: NDArray[np.float64],
    order: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate ``n_steps`` of the order-`order` Adams-Bashforth method.

    The first ``order - 1`` steps are RK4 steps, which supply the past
    slopes the multistep formula needs. Order 1 is explicit Euler.

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
    order : int, default=4
        Number of past slopes used, 1 to 4; also the method's order of
        accuracy.

    Returns
    -------
    times : ndarray of float, shape (n_steps + 1,)
        Time at each step, starting at `t0`.
    states : ndarray of float, shape (n_steps + 1, dim)
        State at each step, starting at `state0`.

    Raises
    ------
    ValueError
        If `order` is not 1, 2, 3 or 4.
    """
    if order < 1 or order > 4:
        raise ValueError("adams_bashforth_integrate: order must be 1, 2, 3 or 4")
    beta = ADAMS_BASHFORTH_COEFFICIENTS[order - 1]
    dim = state0.shape[0]
    states = np.empty((n_steps + 1, dim))
    times = np.empty(n_steps + 1)
    slopes = np.empty((n_steps + 1, dim))
    states[0] = state0
    times[0] = t0
    slopes[0] = rhs(state0, t0, params)
    for i in range(n_steps):
        if i < order - 1:
            state = rk4_step(rhs, states[i], times[i], dt, params)
        else:
            increment = np.zeros(dim)
            for j in range(order):
                increment += beta[j] * slopes[i - j]
            state = states[i] + dt * increment
        states[i + 1] = state
        times[i + 1] = t0 + (i + 1) * dt
        if i + 1 < n_steps:
            slopes[i + 1] = rhs(state, times[i + 1], params)
    return times, states
