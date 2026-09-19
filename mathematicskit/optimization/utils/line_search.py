r"""Backtracking (Armijo) line search -- supporting numerics for
:mod:`mathematicskit.optimization.systems.gradient_descent` and
:mod:`mathematicskit.optimization.systems.conjugate_gradient`, not a model in
its own right. Kept hand-rolled: it is the step-size-selection
*algorithm* these two hand-rolled optimizers demonstrate, not a
numerical primitive with a library replacement.

See Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 3.1,
Algorithm 3.1 ("Backtracking Line Search").
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["backtracking_line_search"]


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    direction: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 50,
) -> float:
    r"""Find a step length satisfying the Armijo sufficient-decrease condition.

    Starting from ``alpha0``, repeatedly shrinks :math:`\alpha \to
    \rho\alpha` until :math:`f(x + \alpha d) \leq f(x) + c_1 \alpha
    \nabla f(x)^T d`, which guarantees the step decreases ``f`` by an
    amount proportional to the step length and the directional
    derivative -- cheap to check (one extra function evaluation per
    trial) and sufficient for global-convergence proofs of the
    descent methods that call it. See Nocedal & Wright, *Numerical
    Optimization*, 2nd ed., Ch. 3.1, Algorithm 3.1.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float``.
    grad : callable
        Gradient ``grad(x) -> ndarray``.
    x : ndarray, shape (n,)
        Current point.
    direction : ndarray, shape (n,)
        Search direction (should be a descent direction, i.e.
        ``grad(x) @ direction < 0``).
    alpha0 : float
        Initial (largest) step length tried.
    c1 : float
        Sufficient-decrease constant, ``0 < c1 < 1`` (standard choice: a
        small value like ``1e-4``).
    rho : float
        Shrink factor per backtrack, ``0 < rho < 1``.
    max_iter : int
        Maximum number of backtracks before giving up and returning the
        smallest step tried.

    Returns
    -------
    float
        A step length ``alpha > 0``.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: x[0] ** 2
    >>> grad = lambda x: np.array([2.0 * x[0]])
    >>> x = np.array([1.0])
    >>> alpha = backtracking_line_search(f, grad, x, direction=-grad(x))
    >>> bool(f(x + alpha * -grad(x)) < f(x))
    True
    """
    fx = f(x)
    slope = grad(x) @ direction
    alpha = float(alpha0)
    for _ in range(max_iter):
        if f(x + alpha * direction) <= fx + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha
