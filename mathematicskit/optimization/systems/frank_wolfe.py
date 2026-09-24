r"""The Frank-Wolfe (conditional gradient) method over a polytope.

Marguerite Frank and Philip Wolfe (1956) minimized a smooth convex
function over a polytope without ever projecting onto it: each step
minimizes the *linearization* of the objective over the feasible set --
a linear program, whose solution :math:`s_k` is a vertex -- and moves
toward it,

.. math::

   s_k = \arg\min_{s \in P} \nabla f(x_k)^T s, \qquad
   x_{k+1} = x_k + \gamma_k (s_k - x_k), \quad \gamma_k = \frac{2}{k+2}.

The *Frank-Wolfe gap* :math:`\nabla f(x_k)^T (x_k - s_k)` bounds
:math:`f(x_k) - f^*` from above, and decreases as :math:`O(1/k)`.
Hand-rolled (SciPy has no conditional-gradient solver); each linear
subproblem is solved with
:func:`~mathematicskit.optimization.systems.linear_programming.linear_program`.
See M. Frank and P. Wolfe, "An Algorithm for Quadratic Programming,"
Naval Research Logistics Quarterly 3(1-2) (1956), 95-110.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER
from mathematicskit.optimization.core.base import OptimizeResult
from mathematicskit.optimization.systems.linear_programming import linear_program

__all__ = ["frank_wolfe"]


def frank_wolfe(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0,
    a_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    a_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds=None,
    tol: float = 1e-6,
    max_iter: int = DEFAULT_MAX_ITER,
) -> OptimizeResult:
    r"""Minimize a smooth convex ``f`` over a bounded polytope by the Frank-Wolfe method.

    The feasible set :math:`P = \{x : A_{ub}x \leq b_{ub},\ A_{eq}x = b_{eq},\ \text{bounds}\}`
    is described exactly as for
    :func:`~mathematicskit.optimization.systems.linear_programming.linear_program`
    (default bounds ``(0, None)``) and must be bounded.

    Parameters
    ----------
    f, grad : callable
        Objective and its gradient.
    x0 : array_like
        A feasible starting point.
    a_ub, b_ub, a_eq, b_eq, bounds
        Polytope description.
    tol : float
        Stop once the Frank-Wolfe gap falls below ``tol``.
    max_iter : int

    Returns
    -------
    OptimizeResult
        ``extra["gaps"]`` holds the Frank-Wolfe gap at every iteration,
        an upper bound on :math:`f(x_k) - f^*`.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.2, 0.3, 0.5])  # already on the probability simplex
    >>> f = lambda x: float(np.sum((x - y) ** 2))
    >>> grad = lambda x: 2.0 * (x - y)
    >>> result = frank_wolfe(f, grad, [1.0, 0.0, 0.0], a_eq=np.ones((1, 3)), b_eq=np.array([1.0]), max_iter=2000)
    >>> np.allclose(result.x, y, atol=1e-2)
    True
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    path = [x.copy()]
    gaps = []
    converged = False
    it = 0
    while it < max_iter:
        g = np.asarray(grad(x), dtype=np.float64)
        lp = linear_program(g, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq, bounds=bounds)
        if not lp.success:
            raise ValueError(f"linear subproblem failed ({lp.message}); is the polytope nonempty and bounded?")
        s = lp.x
        gap = float(g @ (x - s))
        gaps.append(gap)
        if gap < tol:
            converged = True
            break
        gamma = 2.0 / (it + 2.0)
        x = x + gamma * (s - x)
        path.append(x.copy())
        it += 1
    return OptimizeResult(x=x, fun=float(f(x)), path=np.array(path), iterations=it, converged=converged, method="frank_wolfe", extra={"gaps": np.array(gaps)})
