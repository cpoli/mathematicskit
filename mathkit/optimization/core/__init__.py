"""Abstract base classes and result containers for mathkit.optimization."""

from mathkit.optimization.core.base import KKTResult, LagrangeResult, LinearProgramResult, OptimizeResult, UnconstrainedOptimizer

__all__ = ["OptimizeResult", "UnconstrainedOptimizer", "KKTResult", "LagrangeResult", "LinearProgramResult"]
