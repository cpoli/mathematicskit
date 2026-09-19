"""Abstract base classes and result containers for mathematicskit.optimization.

:class:`UnconstrainedOptimizer` is the shared interface for the
iterate-path-exposing methods in :mod:`mathematicskit.optimization.systems`
(gradient descent, nonlinear conjugate gradient, and the scipy-backed
Newton/BFGS wrappers), mirroring
:mod:`mathematicskit.numerical_analysis.core.base.IterativeRootFinder`'s
pattern of one ABC per algorithmic family paired with a stable dataclass
result. :class:`KKTResult` and :class:`LagrangeResult` support the
constrained-optimization utilities in
:mod:`mathematicskit.optimization.systems.constrained`;
:class:`LinearProgramResult` wraps
:func:`~mathematicskit.optimization.systems.linear_programming.linear_program`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL

__all__ = [
    "OptimizeResult",
    "UnconstrainedOptimizer",
    "KKTResult",
    "LagrangeResult",
    "LinearProgramResult",
]


@dataclass
class OptimizeResult:
    """Container for the output of an :class:`UnconstrainedOptimizer`."""

    x: np.ndarray
    """ndarray, shape (n,): The final iterate."""

    fun: float
    """float: Objective value at ``x``."""

    path: np.ndarray
    """ndarray, shape (iterations + 1, n): Every iterate, starting from
    the initial guess -- the record used for convergence-rate comparison
    plots (see :mod:`mathematicskit.optimization.visualizers.plots`)."""

    iterations: int = 0
    """int: Number of iterations actually performed."""

    converged: bool = True
    """bool: Whether the stopping tolerance was met before `max_iter`."""

    method: str = ""
    """str: Name of the method used (e.g. ``"gradient_descent_fixed"``)."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form slot for method-specific diagnostics (e.g. the
    step-size history for a line-search method, or the penalty-parameter
    schedule for :class:`~mathematicskit.optimization.systems.constrained.PenaltyMethod`)."""


class UnconstrainedOptimizer(ABC):
    """Common base for iterative unconstrained-minimization algorithms.

    Parameters
    ----------
    tol : float
        Convergence tolerance on the gradient norm (or, for the
        scipy-backed methods, forwarded as their own tolerance).
    max_iter : int
        Maximum number of iterations before giving up.
    """

    def __init__(self, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    @abstractmethod
    def minimize(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> OptimizeResult:
        """Minimize ``f`` starting from ``x0``, using its gradient ``grad``.

        Parameters
        ----------
        f : callable
            Objective ``f(x) -> float``.
        grad : callable
            Gradient ``grad(x) -> ndarray``, same shape as ``x``.
        x0 : ndarray

        Returns
        -------
        OptimizeResult
        """


@dataclass
class KKTResult:
    """Container for a numerical Karush-Kuhn-Tucker condition check at a candidate point."""

    x: np.ndarray
    """ndarray, shape (n,): The candidate point checked."""

    stationarity_residual: float
    """float: :math:`\\|\\nabla f(x) + \\sum_i \\lambda_i \\nabla h_i(x) +
    \\sum_j \\mu_j \\nabla g_j(x)\\|`, the norm of the (should-be-zero)
    Lagrangian gradient."""

    primal_feasible: bool
    """bool: Whether every equality constraint is (numerically) zero and
    every inequality constraint is :math:`\\leq 0` at ``x``."""

    dual_feasible: bool
    """bool: Whether every inequality multiplier :math:`\\mu_j \\geq 0`."""

    complementary_slackness: bool
    """bool: Whether :math:`\\mu_j g_j(x) \\approx 0` for every inequality constraint."""

    satisfied: bool = False
    """bool: Whether all four KKT conditions above hold within tolerance."""


@dataclass
class LagrangeResult:
    """Container for an equality-constrained stationary point found via
    the Lagrange-multiplier system."""

    x: np.ndarray
    """ndarray, shape (n,): The stationary point found."""

    multipliers: np.ndarray
    """ndarray, shape (m,): The equality-constraint Lagrange multipliers."""

    converged: bool = True
    """bool: Whether the underlying root solve succeeded."""

    kkt: Optional[KKTResult] = None
    """KKTResult, optional: The KKT check at ``(x, multipliers)``, if computed."""


@dataclass
class LinearProgramResult:
    """Container for the output of :func:`~mathematicskit.optimization.systems.linear_programming.linear_program`."""

    x: np.ndarray
    """ndarray, shape (n,): The optimal solution."""

    fun: float
    """float: The optimal objective value."""

    success: bool
    """bool: Whether the solver reported success."""

    message: str = ""
    """str: The underlying solver's status message."""
