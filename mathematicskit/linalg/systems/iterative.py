"""Iterative (Krylov-subspace) linear solvers: conjugate gradient and GMRES.

Thin wrappers around :func:`scipy.sparse.linalg.cg` and
:func:`scipy.sparse.linalg.gmres` -- both are already well-tested,
preconditioner-ready implementations, so mathematicskit does not reimplement the
Krylov-subspace machinery itself. The value mathematicskit adds is the
:class:`~mathematicskit.linalg.core.base.IterativeSolveResult` dataclass and,
via each solver's ``callback``, the *residual history* scipy's functions
don't return on their own (they return only the final iterate), which is
what :mod:`mathematicskit.linalg.visualizers.plots`'
:func:`~mathematicskit.linalg.visualizers.plots.plot_residual_history` plots.
See Golub & Van Loan, *Matrix Computations*, 4th ed., Ch. 11.3 (CG) and
Ch. 9.3.3 (GMRES); Saad, *Iterative Methods for Sparse Linear Systems*,
2nd ed., Ch. 6 (GMRES) and Ch. 6.7 (CG as a special case for SPD systems).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse.linalg as sla

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathematicskit.linalg.core.base import IterativeLinearSolver, IterativeSolveResult

__all__ = ["ConjugateGradient", "GMRES"]


class ConjugateGradient(IterativeLinearSolver):
    r"""Conjugate gradient method for symmetric positive-definite systems.

    Minimizes the quadratic :math:`\phi(x) = \tfrac12 x^T A x - b^T x`
    (equivalent to solving ``A x = b`` for SPD ``A``) by generating
    search directions :math:`A`-conjugate to all previous ones, which
    guarantees exact convergence in at most ``n`` iterations in exact
    arithmetic. Calls :func:`scipy.sparse.linalg.cg` directly. See Golub
    & Van Loan, *Matrix Computations*, 4th ed., Ch. 11.3, Algorithm
    11.3.1.

    Parameters
    ----------
    tol : float
        Relative residual-norm tolerance, ``||r_k|| < tol * ||b||``
        (passed as ``rtol`` to :func:`scipy.sparse.linalg.cg`).
    max_iter : int
        Maximum number of iterations (defaults to
        :data:`~mathematicskit.constants.DEFAULT_MAX_ITER`; ``n`` is normally
        sufficient).

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([1.0, 2.0])
    >>> result = ConjugateGradient().solve(A, b)
    >>> np.allclose(A @ result.x, b, atol=1e-8)
    True
    """

    def __init__(self, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)

    def solve(self, a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> IterativeSolveResult:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = b.shape[0]
        x0_arr = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64)
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-300:
            return IterativeSolveResult(x=x0_arr, residual_history=np.array([0.0]), iterations=0, converged=True, method="conjugate_gradient")

        residuals = [float(np.linalg.norm(b - a @ x0_arr) / b_norm)]

        def callback(xk: np.ndarray) -> None:
            residuals.append(float(np.linalg.norm(b - a @ xk) / b_norm))

        x, info = sla.cg(a, b, x0=x0_arr, rtol=self.tol, atol=0.0, maxiter=self.max_iter, callback=callback)
        iterations = len(residuals) - 1
        converged = info == 0 and residuals[-1] < self.tol
        return IterativeSolveResult(x=x, residual_history=np.array(residuals), iterations=iterations, converged=converged, method="conjugate_gradient")


class GMRES(IterativeLinearSolver):
    r"""GMRES: Generalized Minimal RESidual method for general (non-symmetric) systems.

    Builds an orthonormal basis of the Krylov subspace via the Arnoldi
    process and picks the iterate minimizing the residual norm over it.
    Unlike conjugate gradient, GMRES applies to any nonsingular ``A``
    (not just SPD). Calls :func:`scipy.sparse.linalg.gmres` directly with
    ``restart`` capped at ``n`` (so a single restart cycle already spans
    the full Krylov space, i.e. "full GMRES", matching the textbook
    presentation). See Saad & Schultz (1986), *GMRES: A Generalized
    Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems*,
    SIAM J. Sci. Stat. Comput. 7(3), and Golub & Van Loan, *Matrix
    Computations*, 4th ed., Ch. 9.3.3.

    Parameters
    ----------
    tol : float
        Relative residual-norm tolerance.
    max_iter : int
        Maximum Krylov subspace dimension.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 2.0, 5.0]])
    >>> b = np.array([1.0, 2.0, 3.0])
    >>> result = GMRES().solve(A, b)
    >>> np.allclose(A @ result.x, b, atol=1e-8)
    True
    """

    def __init__(self, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)

    def solve(self, a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> IterativeSolveResult:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = b.shape[0]
        x0_arr = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64)
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-300:
            return IterativeSolveResult(x=x0_arr, residual_history=np.array([0.0]), iterations=0, converged=True, method="gmres")

        residuals = [float(np.linalg.norm(b - a @ x0_arr) / b_norm)]

        def callback(rel_residual: float) -> None:
            residuals.append(float(rel_residual))

        restart = min(self.max_iter, n)
        x, info = sla.gmres(a, b, x0=x0_arr, rtol=self.tol, atol=0.0, restart=restart, maxiter=1, callback=callback, callback_type="pr_norm")
        iterations = len(residuals) - 1
        converged = info == 0 and residuals[-1] < self.tol
        return IterativeSolveResult(x=x, residual_history=np.array(residuals), iterations=iterations, converged=converged, method="gmres")
