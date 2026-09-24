"""Abstract base classes and result containers shared across
mathematicskit.numerical_analysis.

Two families of algorithm live in this domain, so two ABCs cover them
(mirroring physicskit's ``classical/core/base_system.py`` pattern of one
ABC per algorithmic shape, each paired with a stable dataclass result):

* :class:`IterativeRootFinder` -- bisection/Newton/secant/fixed-point:
  algorithms that iterate a scalar sequence toward a root and return a
  :class:`RootResult`.
* :class:`Interpolant` -- Lagrange/Newton-divided-difference/cubic-spline/
  Chebyshev: algorithms that build a function approximating tabulated
  data and are then evaluated at new points via ``evaluate``/``__call__``.

Least-squares polynomial regression (:mod:`mathematicskit.numerical_analysis.systems.regression`)
does not need its own ABC (there is exactly one concrete implementation in
this domain so far); it returns the third dataclass here,
:class:`RegressionResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL

__all__ = [
    "RootResult",
    "IterativeRootFinder",
    "Interpolant",
    "RegressionResult",
    "HornerResult",
    "MinimaxResult",
]


@dataclass
class RootResult:
    """Container for the output of an :class:`IterativeRootFinder`."""

    root: float
    """float: The final estimate of the root."""

    converged: bool
    """bool: Whether the stopping tolerance was met before `max_iter`."""

    iterations: int
    """int: Number of iterations actually performed."""

    history: np.ndarray
    """ndarray, shape (iterations + 1,): The sequence of iterates,
    starting from the initial guess(es), used for convergence-order
    analysis (see :mod:`mathematicskit.numerical_analysis.utils.error_analysis`)."""

    method: str = ""
    """str: Name of the method used (e.g. ``"bisection"``)."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form slot for method-specific diagnostics (e.g. the
    final bracket for bisection)."""


class IterativeRootFinder(ABC):
    """Common base for scalar root-finding algorithms.

    Subclasses store whatever state they need in ``__init__`` (a function,
    a bracket or initial guess(es)) and implement :meth:`solve`.

    Parameters
    ----------
    tol : float
        Convergence tolerance (interpretation is method-specific: an
        absolute change in the iterate, or a bracket width).
    max_iter : int
        Maximum number of iterations before giving up.
    """

    def __init__(self, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    @abstractmethod
    def solve(self) -> RootResult:
        """Run the iteration to convergence (or `max_iter`).

        Returns
        -------
        RootResult
        """


class Interpolant(ABC):
    """Common base for algorithms that build an approximation to
    tabulated data ``(x_i, y_i)`` and can be evaluated at new points.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Interpolation nodes and values, respectively.
    """

    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
        if x.shape[0] < 2:
            raise ValueError("at least 2 nodes are required")
        self.x = x
        self.y = y

    @abstractmethod
    def evaluate(self, x_new):
        """Evaluate the interpolant at new point(s).

        Parameters
        ----------
        x_new : float or array-like of float

        Returns
        -------
        float or ndarray
        """

    def __call__(self, x_new):
        return self.evaluate(x_new)


@dataclass
class RegressionResult:
    """Container for the output of a least-squares regression fit."""

    coefficients: np.ndarray
    """ndarray, shape (degree + 1,): Fitted coefficients, highest power
    first (``numpy.polyval`` convention)."""

    fitted_values: np.ndarray
    """ndarray, shape (n,): Model predictions at the training points."""

    residuals: np.ndarray
    """ndarray, shape (n,): ``y - fitted_values``."""

    r_squared: float
    """float: Coefficient of determination :math:`R^2`."""

    adjusted_r_squared: float
    """float: :math:`R^2` adjusted for the number of predictors."""

    condition_number: Optional[float] = None
    """float, optional: Condition number of the design (Vandermonde)
    matrix used for the fit, a diagnostic for numerical stability at high
    polynomial degree (see :mod:`mathematicskit.numerical_analysis.utils.error_analysis`)."""


@dataclass
class HornerResult:
    r"""Container for one step of Horner's scheme: :math:`p(x_0)` and the
    deflated quotient :math:`q(x)` with :math:`p(x) = (x - x_0)\,q(x) + p(x_0)`."""

    value: float
    """float: :math:`p(x_0)`, the remainder of dividing by :math:`x - x_0`."""

    derivative: float
    """float: :math:`p'(x_0) = q(x_0)`."""

    quotient: np.ndarray
    """ndarray, shape (degree,): Coefficients of :math:`q`, highest power
    first (``numpy.polyval`` convention)."""


@dataclass
class MinimaxResult:
    """Container for the output of the Remez minimax approximation."""

    coefficients: np.ndarray
    """ndarray, shape (degree + 1,): Best-approximation polynomial in the
    power basis, highest power first (``numpy.polyval`` convention)."""

    max_error: float
    r"""float: :math:`\max |f - p|` over the interval, measured on a fine grid."""

    reference: np.ndarray
    """ndarray, shape (degree + 2,): Final reference (alternation) points."""

    iterations: int
    """int: Number of exchange steps performed."""

    converged: bool
    """bool: Whether the error equioscillated to within the tolerance."""

    def evaluate(self, x):
        """Evaluate the minimax polynomial at ``x``."""
        return np.polyval(self.coefficients, x)
