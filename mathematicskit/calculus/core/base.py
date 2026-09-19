"""Abstract base classes and result containers for mathematicskit.calculus.

:class:`Quadrature` is the shared interface for the numerical-integration
rules in :mod:`mathematicskit.calculus.systems.quadrature` (mirroring
:mod:`mathematicskit.numerical_analysis.core.base.IterativeRootFinder`'s pattern
of one ABC per algorithmic family, paired with a stable dataclass
result). Differentiation (:mod:`mathematicskit.calculus.systems.finite_differences`)
and the two automatic-differentiation engines
(:mod:`mathematicskit.calculus.systems.dual_numbers`,
:mod:`mathematicskit.calculus.systems.autodiff`) are simple enough (a handful of
plain functions/small value classes) that a shared ABC would add
ceremony without a real interface to enforce, so they don't get one --
following physicskit's own precedent of varying internal shape by what a
family of algorithms actually needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["QuadratureResult", "Quadrature", "DerivativeResult"]


@dataclass
class QuadratureResult:
    """Container for the output of a :class:`Quadrature` rule."""

    value: float
    """float: Estimated integral."""

    error_estimate: float = 0.0
    """float: Estimated absolute error (0 when the rule has no built-in estimate)."""

    n_evaluations: int = 0
    """int: Number of function evaluations used."""

    method: str = ""
    """str: e.g. ``"trapezoidal"``, ``"simpson"``, ``"gaussian"``, ``"adaptive"``."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics slot (e.g. adaptive quadrature's subinterval history)."""


class Quadrature(ABC):
    """Common base for numerical-integration rules over ``[a, b]``."""

    @abstractmethod
    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        """Estimate :math:`\\int_a^b f(x)\\,dx`.

        Parameters
        ----------
        f : callable
            Integrand ``f(x) -> float``.
        a, b : float
            Integration bounds.

        Returns
        -------
        QuadratureResult
        """


@dataclass
class DerivativeResult:
    """Container for a finite-difference derivative estimate."""

    value: float
    """float: Estimated derivative."""

    step: float = 0.0
    """float: Step size ``h`` used (the finest, for Richardson extrapolation)."""

    method: str = ""
    """str: e.g. ``"forward"``, ``"backward"``, ``"central"``, ``"richardson"``."""

    error_estimate: float = 0.0
    """float: Estimated error, when available (e.g. from Richardson extrapolation's own convergence)."""
