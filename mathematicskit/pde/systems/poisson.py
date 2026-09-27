r"""Poisson :math:`\nabla^2 u = f` and Laplace :math:`\nabla^2 u = 0` equations with Dirichlet data.

The five-point (2D) or three-point (1D) finite-difference discretization
turns the boundary-value problem into a sparse symmetric positive-definite
system :math:`-L_h U = -F + B` (:math:`B` the boundary contribution).
It is solved either directly with :func:`scipy.sparse.linalg.spsolve`, or
iteratively with the :mod:`mathematicskit.linalg` solvers -- conjugate
gradient, Jacobi, Gauss-Seidel, and SOR -- whose residual histories then
show Richardson's (1911) and Young's (1950) relaxation ideas at work. The
iterative solvers operate on dense matrices, so keep those grids small
(a few hundred unknowns).

Laplace (1782) introduced the potential equation for gravitational
attraction outside the masses; Poisson (1813) added the source term for
the potential inside them. See LeVeque, *Finite Difference Methods for
Ordinary and Partial Differential Equations*, SIAM 2007, Ch. 2-4, and
Young, *Iterative Solution of Large Linear Systems*, 1971.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse.linalg as sla

from mathematicskit.constants import DEFAULT_RTOL
from mathematicskit.linalg import SOR, ConjugateGradient, GaussSeidel, JacobiIteration, optimal_sor_omega
from mathematicskit.pde.core.base import EllipticSolution
from mathematicskit.pde.utils.operators import laplacian_1d, laplacian_2d, uniform_grid

__all__ = ["solve_poisson_1d", "solve_poisson_2d", "solve_laplace_2d"]

GridData = float | Callable | np.ndarray


def _on_grid(data: GridData, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if callable(data):
        return np.asarray(data(X, Y), dtype=np.float64) * np.ones(X.shape)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(X.shape, float(arr))
    if arr.shape != X.shape:
        raise ValueError(f"grid data has shape {arr.shape}, expected {X.shape}")
    return arr


def solve_poisson_1d(f: float | Callable | np.ndarray, a: float = 0.0, b: float = 1.0, n: int = 101, ua: float = 0.0, ub: float = 0.0) -> EllipticSolution:
    r"""Solve :math:`u'' = f` on :math:`[a, b]` with :math:`u(a) = u_a`, :math:`u(b) = u_b`.

    Second-order central differences, :math:`O(\Delta x^2)`; the
    tridiagonal system is solved with :func:`scipy.sparse.linalg.spsolve`.

    Parameters
    ----------
    f : float, callable, or ndarray, shape (n,)
        Source term.
    a, b : float
    n : int
        Grid points, both ends included.
    ua, ub : float
        Boundary values.

    Returns
    -------
    EllipticSolution

    Examples
    --------
    >>> import numpy as np
    >>> sol = solve_poisson_1d(2.0, n=11)  # u'' = 2, u(0) = u(1) = 0  =>  u = x^2 - x
    >>> bool(np.allclose(sol.u, sol.x**2 - sol.x))
    True
    """
    x, dx = uniform_grid(a, b, n)
    f_grid = np.asarray(f(x), dtype=np.float64) * np.ones(n) if callable(f) else np.broadcast_to(np.asarray(f, dtype=np.float64), (n,))
    A = -laplacian_1d(n - 2, dx)
    rhs = -f_grid[1:-1].copy()
    rhs[0] += ua / dx**2
    rhs[-1] += ub / dx**2
    interior = sla.spsolve(A.tocsc(), rhs)
    u = np.concatenate([[ua], interior, [ub]])
    return EllipticSolution(x=x, u=u, method="direct", residual_norm=float(np.linalg.norm(A @ interior - rhs)))


def solve_poisson_2d(
    f: GridData,
    n=(33, 33),
    x_range=(0.0, 1.0),
    y_range=(0.0, 1.0),
    boundary: GridData = 0.0,
    method: str = "direct",
    tol: float = DEFAULT_RTOL,
    max_iter: int = 20_000,
    omega=None,
) -> EllipticSolution:
    r"""Solve :math:`u_{xx} + u_{yy} = f` on a rectangle with Dirichlet boundary data.

    Parameters
    ----------
    f : float, callable, or ndarray, shape (nx, ny)
        Source term; a callable is evaluated as ``f(X, Y)`` on ``"ij"`` meshgrids.
    n : tuple of int
        Grid points ``(nx, ny)``, boundaries included.
    x_range, y_range : tuple of float
    boundary : float, callable, or ndarray, shape (nx, ny)
        Dirichlet data; a callable is evaluated as ``g(X, Y)``, and for an
        array only the boundary entries are used.
    method : {"direct", "cg", "jacobi", "gauss_seidel", "sor"}
        ``"direct"`` uses :func:`scipy.sparse.linalg.spsolve`; the others
        use :class:`~mathematicskit.linalg.ConjugateGradient`,
        :class:`~mathematicskit.linalg.JacobiIteration`,
        :class:`~mathematicskit.linalg.GaussSeidel`, and
        :class:`~mathematicskit.linalg.SOR` on the dense matrix.
    tol : float
        Relative residual tolerance for the iterative methods.
    max_iter : int
        Iteration cap for the iterative methods.
    omega : float, optional
        SOR relaxation factor; defaults to
        :func:`~mathematicskit.linalg.optimal_sor_omega` (Young, 1950).

    Returns
    -------
    EllipticSolution
        ``solver_result`` holds the iterative solver's
        :class:`~mathematicskit.linalg.core.base.IterativeSolveResult`.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda X, Y: -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    >>> sol = solve_poisson_2d(f, n=(33, 33))
    >>> X, Y = np.meshgrid(sol.x, sol.y, indexing="ij")
    >>> bool(np.max(np.abs(sol.u - np.sin(np.pi * X) * np.sin(np.pi * Y))) < 1e-2)
    True
    """
    x, dx = uniform_grid(x_range[0], x_range[1], int(n[0]))
    y, dy = uniform_grid(y_range[0], y_range[1], int(n[1]))
    X, Y = np.meshgrid(x, y, indexing="ij")
    f_grid = _on_grid(f, X, Y)
    g = _on_grid(boundary, X, Y).copy()
    g[1:-1, 1:-1] = 0.0
    boundary_term = (g[:-2, 1:-1] + g[2:, 1:-1]) / dx**2 + (g[1:-1, :-2] + g[1:-1, 2:]) / dy**2
    rhs = (boundary_term - f_grid[1:-1, 1:-1]).ravel()
    A = -laplacian_2d(len(x) - 2, len(y) - 2, dx, dy)

    solver_result = None
    if method == "direct":
        interior = sla.spsolve(A.tocsc(), rhs)
    else:
        dense = A.toarray()
        if method == "cg":
            solver = ConjugateGradient(tol=tol, max_iter=max_iter)
        elif method == "jacobi":
            solver = JacobiIteration(tol=tol, max_iter=max_iter)
        elif method == "gauss_seidel":
            solver = GaussSeidel(tol=tol, max_iter=max_iter)
        elif method == "sor":
            solver = SOR(omega=optimal_sor_omega(dense) if omega is None else omega, tol=tol, max_iter=max_iter)
        else:
            raise ValueError(f"Unknown method '{method}'")
        solver_result = solver.solve(dense, rhs)
        interior = solver_result.x

    u = g.copy()
    u[1:-1, 1:-1] = interior.reshape(len(x) - 2, len(y) - 2)
    return EllipticSolution(x=x, y=y, u=u, method=method, residual_norm=float(np.linalg.norm(A @ interior - rhs)), solver_result=solver_result)


def solve_laplace_2d(boundary: GridData, n=(33, 33), x_range=(0.0, 1.0), y_range=(0.0, 1.0), method: str = "direct", **kwargs) -> EllipticSolution:
    r"""Solve Laplace's equation :math:`u_{xx} + u_{yy} = 0` with Dirichlet data `boundary`.

    Equivalent to :func:`solve_poisson_2d` with ``f = 0``; see it for the
    parameters. By the discrete maximum principle, the solution's extreme
    values lie on the boundary.

    Returns
    -------
    EllipticSolution

    Examples
    --------
    >>> import numpy as np
    >>> sol = solve_laplace_2d(lambda X, Y: X + 2 * Y, n=(9, 9))  # harmonic data is reproduced exactly
    >>> X, Y = np.meshgrid(sol.x, sol.y, indexing="ij")
    >>> bool(np.allclose(sol.u, X + 2 * Y))
    True
    """
    return solve_poisson_2d(0.0, n=n, x_range=x_range, y_range=y_range, boundary=boundary, method=method, **kwargs)
