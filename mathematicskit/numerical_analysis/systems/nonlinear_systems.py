r"""Newton's method and Broyden's quasi-Newton method for nonlinear systems
:math:`F(\mathbf x) = \mathbf 0`, :math:`F: \mathbb R^n \to \mathbb R^n`.

Newton's method linearizes :math:`F` at the current iterate and solves
the linear system :math:`J(\mathbf x_k)\,\mathbf s_k = -F(\mathbf x_k)`
for the step; it converges quadratically near a root with nonsingular
Jacobian, but needs :math:`J` at every iterate -- :math:`2n` extra
evaluations of :math:`F` when it is approximated by central differences.
Broyden (1965) builds :math:`J` once and then corrects the approximation
:math:`B_k` after every step by the smallest (rank-one) change that makes
it reproduce the latest secant pair,

.. math::

   B_{k+1} = B_k + \frac{(\mathbf y_k - B_k \mathbf s_k)\,\mathbf s_k^T}{\mathbf s_k^T \mathbf s_k},
   \qquad \mathbf y_k = F(\mathbf x_{k+1}) - F(\mathbf x_k),

trading quadratic for superlinear convergence at one evaluation of
:math:`F` per step. Both are hand-rolled, like the scalar root finders,
because the iterate history is the subject (``scipy.optimize.root``
serves both, and the tests cross-check against it); each linear solve
uses :mod:`mathematicskit.linalg`'s LU factorization. See Burden & Faires,
*Numerical Analysis*, 10th ed., Ch. 10.2-10.3, and C. G. Broyden, "A
Class of Methods for Solving Nonlinear Simultaneous Equations,"
Mathematics of Computation 19 (1965), 577-593.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathematicskit.linalg.systems.lu import lu_decompose, lu_solve
from mathematicskit.numerical_analysis.core.base import SystemRootResult

__all__ = ["numerical_jacobian", "NewtonSystem", "Broyden"]

VectorField = Callable[[np.ndarray], np.ndarray]


def numerical_jacobian(f: VectorField, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector field ``f: R^n -> R^n``.

    Column ``j`` is ``(f(x + h e_j) - f(x - h e_j)) / (2 h)``, accurate to
    :math:`O(h^2)`, at the cost of :math:`2n` evaluations of ``f``.

    Parameters
    ----------
    f : callable
    x : ndarray, shape (n,)
    h : float
        Step size.

    Returns
    -------
    ndarray, shape (n, n)

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: np.array([x[1], -x[0]])
    >>> np.round(numerical_jacobian(f, np.array([0.0, 0.0])), 6)
    array([[ 0.,  1.],
           [-1.,  0.]])
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    jac = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = h
        jac[:, j] = (f(x + dx) - f(x - dx)) / (2.0 * h)
    return jac


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a^{-1} b`` by LU; raises LinAlgError if ``a`` is singular to working precision."""
    return lu_solve(lu_decompose(a), b)


class NewtonSystem:
    r"""Newton's method for :math:`F(\mathbf x) = \mathbf 0`: solve
    :math:`J(\mathbf x_k)\,\mathbf s_k = -F(\mathbf x_k)`, then
    :math:`\mathbf x_{k+1} = \mathbf x_k + \mathbf s_k`.

    Converges quadratically near a root where the Jacobian is nonsingular
    (Burden & Faires, *Numerical Analysis*, 10th ed., Thm. 10.7). The
    Jacobian is the supplied `jacobian`, or :func:`numerical_jacobian`
    otherwise. Iteration stops early, unconverged, if the Jacobian
    becomes singular.

    Parameters
    ----------
    f : callable
        ``F(x) -> ndarray``, same shape as ``x``.
    x0 : array-like, shape (n,)
        Initial guess.
    jacobian : callable, optional
        ``J(x) -> ndarray`` of shape (n, n); finite differences if omitted.
    tol : float
        Stop once ``||x_{k+1} - x_k||_2 < tol``.
    max_iter : int
        Maximum number of iterations.
    h : float
        Finite-difference step when `jacobian` is omitted.

    Examples
    --------
    >>> import numpy as np
    >>> # Intersect the parabola y = x^2 with the line x + y = 2.
    >>> F = lambda v: np.array([v[1] - v[0] ** 2, v[0] + v[1] - 2.0])
    >>> result = NewtonSystem(F, [0.5, 0.5], tol=1e-12).solve()
    >>> np.round(result.root, 10)
    array([1., 1.])
    >>> result.converged
    True
    """

    def __init__(
        self,
        f: VectorField,
        x0,
        jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        tol: float = DEFAULT_RTOL,
        max_iter: int = DEFAULT_MAX_ITER,
        h: float = 1e-6,
    ):
        self.f = f
        self.x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        self.jacobian = jacobian
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.h = float(h)

    def solve(self) -> SystemRootResult:
        x = self.x0.copy()
        n = x.shape[0]
        fx = np.asarray(self.f(x), dtype=np.float64)
        n_eval = 1
        history = [x.copy()]
        norms = [float(np.linalg.norm(fx))]
        converged = False
        n_iter = 0
        extra: dict = {}
        while n_iter < self.max_iter:
            if self.jacobian is not None:
                jac = np.asarray(self.jacobian(x), dtype=np.float64)
            else:
                jac = numerical_jacobian(self.f, x, self.h)
                n_eval += 2 * n
            try:
                step = _solve(jac, -fx)
            except np.linalg.LinAlgError:
                extra["stopped"] = "singular Jacobian"
                break
            n_iter += 1
            x_new = x + step
            fx = np.asarray(self.f(x_new), dtype=np.float64)
            n_eval += 1
            history.append(x_new.copy())
            norms.append(float(np.linalg.norm(fx)))
            done = np.linalg.norm(x_new - x) < self.tol
            x = x_new
            if done:
                converged = True
                break
        return SystemRootResult(
            root=x,
            converged=converged,
            iterations=n_iter,
            history=np.array(history),
            residual_norms=np.array(norms),
            function_evaluations=n_eval,
            method="newton",
            extra=extra,
        )


class Broyden:
    r"""Broyden's ("good") quasi-Newton method for :math:`F(\mathbf x) = \mathbf 0`.

    Starts from :math:`B_0 = J(\mathbf x_0)` and replaces every later
    Jacobian by the rank-one secant update

    .. math::

       B_{k+1} = B_k + \frac{(\mathbf y_k - B_k \mathbf s_k)\,\mathbf s_k^T}{\mathbf s_k^T \mathbf s_k},

    the smallest change (in the Frobenius norm) that satisfies the secant
    condition :math:`B_{k+1}\mathbf s_k = \mathbf y_k`. After the first
    Jacobian, each step costs a single evaluation of :math:`F`, and
    convergence is superlinear (Broyden, Dennis & Moré 1973). See
    Broyden, Mathematics of Computation 19 (1965), 577-593, and Burden &
    Faires, *Numerical Analysis*, 10th ed., Ch. 10.3.

    Parameters
    ----------
    f : callable
        ``F(x) -> ndarray``, same shape as ``x``.
    x0 : array-like, shape (n,)
        Initial guess.
    jacobian : callable, optional
        ``J(x) -> ndarray``, evaluated once at `x0` for :math:`B_0`;
        finite differences if omitted.
    tol : float
        Stop once ``||x_{k+1} - x_k||_2 < tol``.
    max_iter : int
        Maximum number of iterations.
    h : float
        Finite-difference step when `jacobian` is omitted.

    Examples
    --------
    >>> import numpy as np
    >>> F = lambda v: np.array([v[1] - v[0] ** 2, v[0] + v[1] - 2.0])
    >>> result = Broyden(F, [0.8, 0.8], tol=1e-12).solve()
    >>> np.round(result.root, 10)
    array([1., 1.])
    >>> # One F evaluation per step after the initial Jacobian (5 evaluations).
    >>> result.function_evaluations == 1 + 4 + result.iterations
    True
    """

    def __init__(
        self,
        f: VectorField,
        x0,
        jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        tol: float = DEFAULT_RTOL,
        max_iter: int = DEFAULT_MAX_ITER,
        h: float = 1e-6,
    ):
        self.f = f
        self.x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        self.jacobian = jacobian
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.h = float(h)

    def solve(self) -> SystemRootResult:
        x = self.x0.copy()
        n = x.shape[0]
        fx = np.asarray(self.f(x), dtype=np.float64)
        n_eval = 1
        if self.jacobian is not None:
            B = np.asarray(self.jacobian(x), dtype=np.float64).copy()
        else:
            B = numerical_jacobian(self.f, x, self.h)
            n_eval += 2 * n
        history = [x.copy()]
        norms = [float(np.linalg.norm(fx))]
        converged = False
        n_iter = 0
        extra: dict = {}
        while n_iter < self.max_iter:
            try:
                step = _solve(B, -fx)
            except np.linalg.LinAlgError:
                extra["stopped"] = "singular Jacobian approximation"
                break
            n_iter += 1
            x_new = x + step
            f_new = np.asarray(self.f(x_new), dtype=np.float64)
            n_eval += 1
            history.append(x_new.copy())
            norms.append(float(np.linalg.norm(f_new)))
            step_sq = float(step @ step)
            if step_sq > 0.0:
                B += np.outer(f_new - fx - B @ step, step) / step_sq
            x, fx = x_new, f_new
            if np.sqrt(step_sq) < self.tol:
                converged = True
                break
        extra["jacobian_approximation"] = B
        return SystemRootResult(
            root=x,
            converged=converged,
            iterations=n_iter,
            history=np.array(history),
            residual_norms=np.array(norms),
            function_evaluations=n_eval,
            method="broyden",
            extra=extra,
        )
