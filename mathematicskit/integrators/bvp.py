"""Two-point boundary-value problems by collocation.

:func:`collocation_bvp` wraps :func:`scipy.integrate.solve_bvp`, a 4th-order
collocation method (3-stage Lobatto IIIA, equivalently a cubic
C1-continuous spline satisfying the ODE at each interval's endpoints and
midpoint) with residual-based mesh refinement (Kierzenka & Shampine 2001).
Unlike an initial-value integrator, the whole solution is found at once:
the collocation conditions and boundary conditions form one nonlinear
system solved by damped Newton iterations, so a nonlinear BVP may have
several solutions, and which one is found depends on the initial guess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_bvp

from mathematicskit.integrators.fixed_step import RHSFunc

__all__ = ["BVPResult", "collocation_bvp"]

#: Boundary-condition residual ``bc(state_a, state_b, params) -> ndarray``,
#: zero when the solution's endpoint states satisfy the conditions.
BCFunc = Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], ArrayLike]


@dataclass
class BVPResult:
    """Solution of a two-point boundary-value problem (see :func:`collocation_bvp`)."""

    x: NDArray[np.float64]
    """ndarray, shape (n_nodes,): Final (refined) mesh."""

    y: NDArray[np.float64]
    """ndarray, shape (n_nodes, dim): Solution at the mesh nodes."""

    sol: Callable[[ArrayLike], NDArray[np.float64]]
    """callable: ``sol(x) -> ndarray (len(x), dim)``, the continuous
    cubic-spline solution evaluated anywhere in the interval."""

    success: bool
    """bool: Whether the requested tolerance was met."""

    message: str
    """str: Solver termination message."""

    max_rms_residual: float
    """float: Largest per-interval RMS relative residual of the collocation spline."""


def collocation_bvp(
    rhs: RHSFunc,
    bc: BCFunc,
    x: ArrayLike,
    state_guess: ArrayLike,
    params: NDArray[np.float64],
    tol: float = 1e-6,
    max_nodes: int = 10_000,
) -> BVPResult:
    """Solve ``state' = rhs(state, x, params)`` subject to ``bc(state(a), state(b)) = 0``.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(state, x, params) -> ndarray (dim,)``, in the
        same convention as the initial-value integrators (an ``@njit``
        function or a plain Python one).
    bc : callable
        Boundary residual ``bc(state_a, state_b, params) -> ndarray (dim,)``.
    x : array_like of float, shape (m,)
        Initial mesh, strictly increasing, from ``a`` to ``b``.
    state_guess : array_like of float, shape (m, dim)
        Initial guess for the solution at each mesh node.
    params : ndarray of float
        Parameter vector passed through to `rhs` and `bc`.
    tol : float, default=1e-6
        Target relative collocation residual.
    max_nodes : int, default=10000
        Mesh-refinement cap.

    Returns
    -------
    BVPResult
        The solution; check ``success`` before trusting it.

    Examples
    --------
    ``y'' = -y`` with ``y(0) = 0``, ``y(pi/2) = 1`` has solution ``sin x``:

    >>> import numpy as np
    >>> def rhs(state, x, params):
    ...     return np.array([state[1], -state[0]])
    >>> def bc(ya, yb, params):
    ...     return np.array([ya[0], yb[0] - 1.0])
    >>> x = np.linspace(0.0, np.pi / 2, 11)
    >>> res = collocation_bvp(rhs, bc, x, np.zeros((11, 2)), np.zeros(0))
    >>> res.success, bool(np.allclose(res.sol([0.5])[0, 0], np.sin(0.5), atol=1e-6))
    (True, True)
    """

    def fun(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.float64]:
        rows = np.ascontiguousarray(ys.T)
        return np.column_stack([rhs(rows[i], xs[i], params) for i in range(xs.size)])

    sol = solve_bvp(
        fun,
        lambda ya, yb: np.asarray(bc(ya, yb, params), dtype=np.float64),
        np.asarray(x, dtype=np.float64),
        np.asarray(state_guess, dtype=np.float64).T,
        tol=tol,
        max_nodes=max_nodes,
    )
    return BVPResult(
        x=sol.x,
        y=sol.y.T,
        sol=lambda xq: sol.sol(np.atleast_1d(np.asarray(xq, dtype=np.float64))).T,
        success=bool(sol.success),
        message=sol.message,
        max_rms_residual=float(np.max(sol.rms_residuals)),
    )
