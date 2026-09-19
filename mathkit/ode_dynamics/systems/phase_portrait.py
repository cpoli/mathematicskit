r"""2D flow systems and phase-portrait grid generation.

:class:`Linear2D` and :class:`Nonlinear2D` wrap arbitrary planar vector
fields as :class:`~mathkit.ode_dynamics.core.base.FlowSystem`\ s (built
on :mod:`mathkit.integrators`); :func:`vector_field_grid` samples a
vector field on a grid for quiver-style phase-portrait plots.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from numba import njit

from mathkit.ode_dynamics.core.base import FlowSystem

__all__ = ["Linear2D", "Nonlinear2D", "vector_field_grid"]


@lru_cache(maxsize=64)
def _make_linear_rhs(a11: float, a12: float, a21: float, a22: float):
    @njit(cache=True)
    def rhs(state, t, params):
        x, y = state[0], state[1]
        out = np.empty(2)
        out[0] = a11 * x + a12 * y
        out[1] = a21 * x + a22 * y
        return out

    return rhs


class Linear2D(FlowSystem):
    r"""Linear planar system :math:`\dot{\mathbf x} = A\mathbf x`.

    The textbook object of study for
    :func:`mathkit.ode_dynamics.systems.stability.classify_fixed_point_2d`:
    every trajectory's qualitative behavior near the origin is fully
    determined by ``A``'s eigenvalues. See Strogatz, *Nonlinear Dynamics
    and Chaos*, 2nd ed., Ch. 5.

    Parameters
    ----------
    state0 : array-like, shape (2,)
        Initial condition.
    A : array-like, shape (2, 2)
        System matrix.

    Examples
    --------
    >>> import numpy as np
    >>> system = Linear2D([1.0, 0.0], A=[[-1.0, 0.0], [0.0, -2.0]])
    >>> result = system.integrate((0.0, 5.0), dt=1e-3, method="rk4")
    >>> bool(abs(result.y[-1, 0]) < 1e-2)
    True
    """

    def __init__(self, state0, A):
        A = np.asarray(A, dtype=np.float64)
        if A.shape != (2, 2):
            raise ValueError("A must be a 2x2 matrix")
        self.A = A
        self._rhs_njit = _make_linear_rhs(float(A[0, 0]), float(A[0, 1]), float(A[1, 0]), float(A[1, 1]))
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        return self.A @ state


class Nonlinear2D(FlowSystem):
    r"""General nonlinear planar system :math:`\dot{\mathbf x} = f(\mathbf x)`.

    Wraps an arbitrary plain-Python vector field for integration and
    phase-portrait visualization (a compiled njit dispatcher is built
    lazily via numba's dynamic ``njit`` on the supplied callable, so any
    numba-compatible pure function works).

    Parameters
    ----------
    state0 : array-like, shape (2,)
    f : callable
        ``f(x, y) -> (dx, dy)``, numba-compatible (no Python objects).

    Examples
    --------
    >>> import numpy as np
    >>> vdp = Nonlinear2D([2.0, 0.0], f=lambda x, y: (y, -x + 0.0 * y))
    >>> result = vdp.integrate((0.0, 1.0), dt=1e-3, method="rk4")
    >>> result.y.shape[1]
    2
    """

    def __init__(self, state0, f: Callable[[float, float], tuple]):
        self._f_plain = f
        compiled = njit(cache=False)(f)

        @njit(cache=False)
        def rhs(state, t, params):
            dx, dy = compiled(state[0], state[1])
            out = np.empty(2)
            out[0] = dx
            out[1] = dy
            return out

        self._rhs_njit = rhs
        self.params = np.empty(0)
        super().__init__(state0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        dx, dy = self._f_plain(state[0], state[1])
        return np.array([dx, dy])


def vector_field_grid(f: Callable[[float, float], tuple], x_range, y_range, n: int = 20):
    """Sample a planar vector field on an ``n x n`` grid.

    Parameters
    ----------
    f : callable
        ``f(x, y) -> (dx, dy)`` (plain Python; not required to be
        numba-compatible, unlike :class:`Nonlinear2D`'s constructor).
    x_range, y_range : tuple of float
        ``(min, max)`` for each axis.
    n : int
        Grid resolution per axis.

    Returns
    -------
    X, Y, U, V : ndarray, shape (n, n)
        Grid coordinates and vector-field components, suitable for
        ``matplotlib.pyplot.quiver(X, Y, U, V)``.

    Examples
    --------
    >>> X, Y, U, V = vector_field_grid(lambda x, y: (y, -x), (-1, 1), (-1, 1), n=5)
    >>> X.shape
    (5, 5)
    """
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(n):
        for j in range(n):
            u, v = f(X[i, j], Y[i, j])
            U[i, j] = u
            V[i, j] = v
    return X, Y, U, V
