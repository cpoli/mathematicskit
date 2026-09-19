r"""Nonlinear conjugate gradient (Fletcher-Reeves and Polak-Ribiere variants).

Hand-rolled, like :mod:`mathematicskit.optimization.systems.gradient_descent`,
to expose the per-iterate path. Nonlinear CG is the natural next step up
from steepest descent: it reuses previous search directions to avoid
gradient descent's characteristic zig-zagging in narrow valleys, without
needing second-derivative information the way Newton's method does. See
Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 5.2.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathematicskit.optimization.core.base import OptimizeResult, UnconstrainedOptimizer
from mathematicskit.optimization.utils.line_search import backtracking_line_search

__all__ = ["NonlinearConjugateGradient"]


class NonlinearConjugateGradient(UnconstrainedOptimizer):
    r"""Nonlinear conjugate gradient with a backtracking line search.

    Generates search directions :math:`d_{k+1} = -\nabla f(x_{k+1}) +
    \beta_k d_k`, restarting at :math:`d_0 = -\nabla f(x_0)`. Two
    classic choices of :math:`\beta_k`:

    - **Fletcher-Reeves**: :math:`\beta_k^{FR} = \dfrac{\nabla
      f(x_{k+1})^T \nabla f(x_{k+1})}{\nabla f(x_k)^T \nabla f(x_k)}`.
    - **Polak-Ribiere** (the default, with the standard
      :math:`\max(0, \cdot)` safeguard that automatically restarts along
      the steepest-descent direction if :math:`\beta_k` would go
      negative): :math:`\beta_k^{PR+} = \max\!\left(0,
      \dfrac{\nabla f(x_{k+1})^T(\nabla f(x_{k+1}) - \nabla
      f(x_k))}{\nabla f(x_k)^T \nabla f(x_k)}\right)`.

    See Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 5.2,
    eq. (5.41)-(5.45).

    Parameters
    ----------
    variant : {"polak_ribiere", "fletcher_reeves"}
    c1, rho, alpha0 : float
        Forwarded to :func:`~mathematicskit.optimization.utils.line_search.backtracking_line_search`.
    tol : float
        Convergence tolerance on ``||grad(x)||``.
    max_iter : int

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: x[0] ** 2 + 10.0 * x[1] ** 2
    >>> grad = lambda x: np.array([2.0 * x[0], 20.0 * x[1]])
    >>> result = NonlinearConjugateGradient(tol=1e-10).minimize(f, grad, np.array([3.0, 4.0]))
    >>> np.allclose(result.x, [0.0, 0.0], atol=1e-6)
    True
    """

    def __init__(
        self,
        variant: str = "polak_ribiere",
        c1: float = 1e-4,
        rho: float = 0.5,
        alpha0: float = 1.0,
        tol: float = DEFAULT_RTOL,
        max_iter: int = DEFAULT_MAX_ITER,
    ):
        super().__init__(tol=tol, max_iter=max_iter)
        if variant not in ("polak_ribiere", "fletcher_reeves"):
            raise ValueError(f"variant must be 'polak_ribiere' or 'fletcher_reeves', got {variant!r}")
        self.variant = variant
        self.c1, self.rho, self.alpha0 = float(c1), float(rho), float(alpha0)

    def minimize(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> OptimizeResult:
        x = np.asarray(x0, dtype=np.float64).copy()
        g = grad(x)
        d = -g
        path = [x.copy()]
        converged = False
        it = 0
        while it < self.max_iter:
            if np.linalg.norm(g) < self.tol:
                converged = True
                break
            it += 1
            alpha = backtracking_line_search(f, grad, x, d, alpha0=self.alpha0, c1=self.c1, rho=self.rho)
            x_new = x + alpha * d
            g_new = grad(x_new)
            path.append(x_new.copy())

            gg_old = g @ g
            if gg_old < 1e-300:
                beta = 0.0
            elif self.variant == "fletcher_reeves":
                beta = float(g_new @ g_new) / gg_old
            else:
                beta = max(0.0, float(g_new @ (g_new - g)) / gg_old)

            d = -g_new + beta * d
            x, g = x_new, g_new

        return OptimizeResult(x=x, fun=float(f(x)), path=np.array(path), iterations=it, converged=converged, method=f"nonlinear_cg_{self.variant}")
