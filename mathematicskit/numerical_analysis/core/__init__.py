"""Abstract base classes and result containers for mathematicskit.numerical_analysis."""

from mathematicskit.numerical_analysis.core.base import (
    Interpolant,
    IterativeRootFinder,
    RegressionResult,
    RootResult,
)

__all__ = [
    "RootResult",
    "IterativeRootFinder",
    "Interpolant",
    "RegressionResult",
]
