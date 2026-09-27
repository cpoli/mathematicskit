r"""Geometric multigrid V-cycles for the 2D Poisson equation.

Relaxation (Jacobi, Gauss-Seidel) removes *oscillatory* error in a few
sweeps but *smooth* error only very slowly, which is why its iteration
count grows with the grid size. Fedorenko (1964) and Brandt (1977)
observed that smooth error on a fine grid is oscillatory on a coarser
one: smooth, restrict the residual to a grid with twice the spacing,
solve for the correction there (recursively), interpolate it back, and
smooth again. One such V-cycle cuts the error by a fixed factor
(about 0.2 here) *independent of the grid size*, so the total work is
:math:`O(N)` in the number of unknowns.

This is a hand-rolled textbook V(:math:`\nu_1`, :math:`\nu_2`) cycle --
weighted-Jacobi smoothing (:math:`\omega = 4/5`), full-weighting
restriction, bilinear interpolation -- on the unit-spaced square grids
:math:`n = 2^k + 1` (scipy has no multigrid; the cycle is the
pedagogical subject). See Briggs, Henson & McCormick, *A Multigrid
Tutorial*, 2nd ed., SIAM 2000, Ch. 3-4.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.linalg.core.base import IterativeSolveResult
from mathematicskit.pde.core.base import EllipticSolution
from mathematicskit.pde.systems.poisson import GridData, _on_grid
from mathematicskit.pde.utils.operators import uniform_grid

__all__ = ["multigrid_poisson_2d"]


def _laplacian(u: np.ndarray, h: float) -> np.ndarray:
    return (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]) / h**2


def _residual(u: np.ndarray, f: np.ndarray, h: float) -> np.ndarray:
    r = np.zeros_like(u)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - _laplacian(u, h)
    return r


def _smooth(u: np.ndarray, f: np.ndarray, h: float, sweeps: int, omega: float) -> np.ndarray:
    for _ in range(sweeps):
        jacobi = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - h**2 * f[1:-1, 1:-1])
        u[1:-1, 1:-1] = (1.0 - omega) * u[1:-1, 1:-1] + omega * jacobi
    return u


def _restrict(r: np.ndarray) -> np.ndarray:
    """Full weighting: stencil [1 2 1; 2 4 2; 1 2 1] / 16 at every other point."""
    coarse = np.zeros(((r.shape[0] + 1) // 2, (r.shape[1] + 1) // 2))
    c = r[2:-2:2, 2:-2:2]
    edges = r[1:-3:2, 2:-2:2] + r[3:-1:2, 2:-2:2] + r[2:-2:2, 1:-3:2] + r[2:-2:2, 3:-1:2]
    corners = r[1:-3:2, 1:-3:2] + r[1:-3:2, 3:-1:2] + r[3:-1:2, 1:-3:2] + r[3:-1:2, 3:-1:2]
    coarse[1:-1, 1:-1] = (4.0 * c + 2.0 * edges + corners) / 16.0
    return coarse


def _prolong(e: np.ndarray) -> np.ndarray:
    """Bilinear interpolation from a coarse grid to the grid with half its spacing."""
    fine = np.zeros((2 * e.shape[0] - 1, 2 * e.shape[1] - 1))
    fine[::2, ::2] = e
    fine[1::2, ::2] = 0.5 * (e[:-1, :] + e[1:, :])
    fine[::2, 1::2] = 0.5 * (e[:, :-1] + e[:, 1:])
    fine[1::2, 1::2] = 0.25 * (e[:-1, :-1] + e[1:, :-1] + e[:-1, 1:] + e[1:, 1:])
    return fine


def _v_cycle(u: np.ndarray, f: np.ndarray, h: float, pre: int, post: int, omega: float) -> np.ndarray:
    if u.shape[0] <= 3:  # one interior point: solve exactly
        u[1, 1] = 0.25 * (u[0, 1] + u[2, 1] + u[1, 0] + u[1, 2] - h**2 * f[1, 1])
        return u
    u = _smooth(u, f, h, pre, omega)
    coarse_residual = _restrict(_residual(u, f, h))
    correction = _v_cycle(np.zeros_like(coarse_residual), coarse_residual, 2.0 * h, pre, post, omega)
    u += _prolong(correction)
    return _smooth(u, f, h, post, omega)


def multigrid_poisson_2d(
    f: GridData,
    n: int = 65,
    boundary: GridData = 0.0,
    tol: float = 1e-8,
    max_cycles: int = 50,
    pre_smooth: int = 2,
    post_smooth: int = 2,
    omega: float = 0.8,
) -> EllipticSolution:
    r"""Solve :math:`u_{xx} + u_{yy} = f` on the unit square by multigrid V-cycles.

    Discretization and boundary handling match
    :func:`~mathematicskit.pde.systems.poisson.solve_poisson_2d` (five-point
    Laplacian, Dirichlet data), so both converge to the same discrete solution.

    Parameters
    ----------
    f : float, callable, or ndarray, shape (n, n)
        Source term; a callable is evaluated as ``f(X, Y)`` on ``"ij"`` meshgrids.
    n : int
        Grid points per side, boundaries included; must be ``2**k + 1``.
    boundary : float, callable, or ndarray, shape (n, n)
        Dirichlet data.
    tol : float
        Stop when the residual 2-norm falls below `tol` times its initial value.
    max_cycles : int
    pre_smooth, post_smooth : int
        Weighted-Jacobi sweeps before and after each coarse-grid correction.
    omega : float
        Jacobi damping (4/5 is optimal for smoothing the 2D five-point Laplacian).

    Returns
    -------
    EllipticSolution
        ``solver_result`` is an
        :class:`~mathematicskit.linalg.core.base.IterativeSolveResult` whose
        ``residual_history`` holds the relative residual after each V-cycle.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda X, Y: -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    >>> sol = multigrid_poisson_2d(f, n=33)
    >>> sol.solver_result.converged, sol.solver_result.iterations < 15
    (True, True)
    """
    if n < 3 or (n - 1) & (n - 2):
        raise ValueError("n must be 2**k + 1 (e.g. 17, 33, 65, 129)")
    x, h = uniform_grid(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f_grid = _on_grid(f, X, Y)
    u = _on_grid(boundary, X, Y).copy()
    u[1:-1, 1:-1] = 0.0
    r0 = np.linalg.norm(_residual(u, f_grid, h))
    history = [1.0]
    converged = r0 == 0.0
    cycles = 0
    while not converged and cycles < max_cycles:
        u = _v_cycle(u, f_grid, h, pre_smooth, post_smooth, omega)
        cycles += 1
        history.append(float(np.linalg.norm(_residual(u, f_grid, h)) / r0))
        converged = history[-1] < tol
    result = IterativeSolveResult(x=u[1:-1, 1:-1].ravel(), residual_history=np.array(history), iterations=cycles, converged=converged, method="multigrid")
    return EllipticSolution(x=x, y=x.copy(), u=u, method="multigrid", residual_norm=float(history[-1] * r0), solver_result=result)
