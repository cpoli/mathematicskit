"""Abstract base classes and result containers for mathkit.numerical_analysis."""

from mathkit.numerical_analysis.core.base import (
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
