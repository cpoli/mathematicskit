"""Convergence-rate comparison utilities across optimization methods on a
shared test function -- supporting numerics for the examples gallery,
not a model in its own right.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathkit.optimization.core.base import OptimizeResult, UnconstrainedOptimizer

__all__ = ["compare_optimizers", "function_value_gap"]


def compare_optimizers(
    optimizers: dict[str, UnconstrainedOptimizer], f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray
) -> dict[str, OptimizeResult]:
    """Run several optimizers from the same starting point on the same problem.

    Parameters
    ----------
    optimizers : dict of str -> UnconstrainedOptimizer
        Method name -> configured optimizer instance.
    f : callable
    grad : callable
    x0 : ndarray

    Returns
    -------
    dict of str -> OptimizeResult

    Examples
    --------
    >>> import numpy as np
    >>> from mathkit.optimization.systems.gradient_descent import GradientDescent, GradientDescentLineSearch
    >>> f = lambda x: x[0] ** 2 + x[1] ** 2
    >>> grad = lambda x: np.array([2.0 * x[0], 2.0 * x[1]])
    >>> results = compare_optimizers({"fixed": GradientDescent(alpha=0.1), "line_search": GradientDescentLineSearch()}, f, grad, np.array([3.0, 4.0]))
    >>> sorted(results.keys())
    ['fixed', 'line_search']
    """
    return {name: opt.minimize(f, grad, x0) for name, opt in optimizers.items()}


def function_value_gap(result: OptimizeResult, f: Callable[[np.ndarray], float], f_star: float) -> np.ndarray:
    """Compute ``f(x_k) - f*`` at every recorded iterate -- the standard
    y-axis for a convergence-rate plot (log-scale reveals linear vs.
    superlinear vs. quadratic convergence as a straight, curving-down, or
    sharply-curving-down line, respectively).

    Parameters
    ----------
    result : OptimizeResult
        From any :class:`~mathkit.optimization.core.base.UnconstrainedOptimizer`.
    f : callable
    f_star : float
        The known (or best available) optimal value.

    Returns
    -------
    ndarray, shape (iterations + 1,)

    Examples
    --------
    >>> import numpy as np
    >>> from mathkit.optimization.systems.gradient_descent import GradientDescent
    >>> f = lambda x: x[0] ** 2
    >>> grad = lambda x: np.array([2.0 * x[0]])
    >>> result = GradientDescent(alpha=0.1, tol=1e-10).minimize(f, grad, np.array([1.0]))
    >>> gap = function_value_gap(result, f, f_star=0.0)
    >>> bool(np.all(np.diff(gap) <= 1e-12))
    True
    """
    return np.array([f(x) - f_star for x in result.path])
