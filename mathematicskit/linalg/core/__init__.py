"""Abstract base classes and result containers for mathematicskit.linalg."""

from mathematicskit.linalg.core.base import (
    CholeskyResult,
    EigenResult,
    IterativeLinearSolver,
    IterativeSolveResult,
    LeastSquaresResult,
    LUResult,
    QRResult,
    SVDResult,
)

__all__ = [
    "LUResult",
    "QRResult",
    "CholeskyResult",
    "EigenResult",
    "SVDResult",
    "IterativeSolveResult",
    "LeastSquaresResult",
    "IterativeLinearSolver",
]
