r"""Gradient descent, with fixed and line-search step sizes.

Hand-rolled to expose the full per-iterate path used by the
convergence-rate comparisons in
:mod:`mathkit.optimization.utils.comparison` -- ``scipy.optimize``'s
solvers don't return that history. See Nocedal & Wright, *Numerical
Optimization*, 2nd ed., Ch. 2.2 and Ch. 3.1.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathkit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathkit.optimization.core.base import OptimizeResult, UnconstrainedOptimizer
from mathkit.optimization.utils.line_search import backtracking_line_search

__all__ = ["GradientDescent", "GradientDescentLineSearch"]


class GradientDescent(UnconstrainedOptimizer):
    r"""Fixed-step gradient descent: :math:`x_{k+1} = x_k - \alpha \nabla f(x_k)`.

    The simplest descent method: always steps a constant multiple
    ``alpha`` of the negative gradient. Converges linearly for
    ``alpha`` small enough (``alpha < 2/L`` for ``L``-smooth convex
    ``f``), but a poor choice of ``alpha`` either diverges (too large) or
    converges glacially (too small) -- unlike
    :class:`GradientDescentLineSearch`, which adapts the step
    automatically. See Nocedal & Wright, *Numerical Optimization*, 2nd
    ed., Ch. 2.2.

    Parameters
    ----------
    alpha : float
        Fixed step size.
    tol : float
        Convergence tolerance on ``||grad(x)||``.
    max_iter : int

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: x[0] ** 2 + x[1] ** 2
    >>> grad = lambda x: np.array([2.0 * x[0], 2.0 * x[1]])
    >>> result = GradientDescent(alpha=0.1, tol=1e-10).minimize(f, grad, np.array([3.0, 4.0]))
    >>> np.allclose(result.x, [0.0, 0.0], atol=1e-6)
    True
    """

    def __init__(self, alpha: float = 0.1, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        self.alpha = float(alpha)

    def minimize(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> OptimizeResult:
        x = np.asarray(x0, dtype=np.float64).copy()
        path = [x.copy()]
        converged = False
        it = 0
        while it < self.max_iter:
            g = grad(x)
            if np.linalg.norm(g) < self.tol:
                converged = True
                break
            it += 1
            x = x - self.alpha * g
            path.append(x.copy())
        return OptimizeResult(x=x, fun=float(f(x)), path=np.array(path), iterations=it, converged=converged, method="gradient_descent_fixed")


class GradientDescentLineSearch(UnconstrainedOptimizer):
    r"""Gradient descent with backtracking (Armijo) line search.

    :math:`x_{k+1} = x_k + \alpha_k d_k`, :math:`d_k = -\nabla f(x_k)`,
    with :math:`\alpha_k` chosen each step by
    :func:`~mathkit.optimization.utils.line_search.backtracking_line_search`
    rather than fixed in advance -- robust to poor scaling that would
    make a fixed step size either diverge or crawl. See Nocedal & Wright,
    *Numerical Optimization*, 2nd ed., Ch. 3.1.

    Parameters
    ----------
    c1, rho : float
        Armijo sufficient-decrease constant and backtrack shrink factor
        (forwarded to :func:`~mathkit.optimization.utils.line_search.backtracking_line_search`).
    alpha0 : float
        Initial step length tried at each iteration.
    tol : float
        Convergence tolerance on ``||grad(x)||``.
    max_iter : int

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: x[0] ** 2 + 10.0 * x[1] ** 2
    >>> grad = lambda x: np.array([2.0 * x[0], 20.0 * x[1]])
    >>> result = GradientDescentLineSearch(tol=1e-10).minimize(f, grad, np.array([3.0, 4.0]))
    >>> np.allclose(result.x, [0.0, 0.0], atol=1e-6)
    True
    """

    def __init__(self, c1: float = 1e-4, rho: float = 0.5, alpha0: float = 1.0, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        self.c1 = float(c1)
        self.rho = float(rho)
        self.alpha0 = float(alpha0)

    def minimize(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> OptimizeResult:
        x = np.asarray(x0, dtype=np.float64).copy()
        path = [x.copy()]
        step_sizes = []
        converged = False
        it = 0
        while it < self.max_iter:
            g = grad(x)
            if np.linalg.norm(g) < self.tol:
                converged = True
                break
            it += 1
            d = -g
            alpha = backtracking_line_search(f, grad, x, d, alpha0=self.alpha0, c1=self.c1, rho=self.rho)
            x = x + alpha * d
            path.append(x.copy())
            step_sizes.append(alpha)
        return OptimizeResult(
            x=x,
            fun=float(f(x)),
            path=np.array(path),
            iterations=it,
            converged=converged,
            method="gradient_descent_line_search",
            extra={"step_sizes": np.array(step_sizes)},
        )
