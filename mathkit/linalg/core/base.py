"""Abstract base classes and result containers for mathkit.linalg.

Every decomposition/solver in this domain returns a small dataclass
(mirroring physicskit's ``SimulationResult`` pattern). Most of this
domain's algorithms call ``numpy.linalg``/``scipy.linalg``/
``scipy.sparse.linalg`` directly (see each ``systems/`` module's
docstring for the specific library routine); the Krylov-subspace linear
solvers (:mod:`mathkit.linalg.systems.iterative`) share the
:class:`IterativeLinearSolver` ABC, the same shape as
:mod:`mathkit.numerical_analysis.core.base.IterativeRootFinder`, purely
to expose a stable ``.solve()`` interface and residual-history tracking
around ``scipy.sparse.linalg.cg``/``gmres`` -- the ABC is not itself a
hand-rolled numerical algorithm. Power iteration, inverse iteration, and
the Gram-Schmidt-vs-Householder QR comparison remain hand-rolled in
their respective modules because the iteration/comparison itself is the
pedagogical content, not something a library call would expose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mathkit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL

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


@dataclass
class LUResult:
    """Output of an LU decomposition with partial pivoting: ``P A = L U``."""

    L: np.ndarray
    """ndarray, shape (n, n): Unit lower-triangular factor."""

    U: np.ndarray
    """ndarray, shape (n, n): Upper-triangular factor."""

    P: np.ndarray
    """ndarray, shape (n, n): Row-permutation matrix, ``P @ A == L @ U``."""

    num_row_swaps: int = 0
    """int: Number of row swaps performed (parity gives ``det(P)``)."""


@dataclass
class QRResult:
    """Output of a QR decomposition: ``A = Q R``."""

    Q: np.ndarray
    """ndarray, shape (m, n): Orthonormal-columns factor."""

    R: np.ndarray
    """ndarray, shape (n, n): Upper-triangular factor."""

    method: str = ""
    """str: ``"householder"`` or ``"gram_schmidt"``."""


@dataclass
class CholeskyResult:
    """Output of a Cholesky decomposition of an SPD matrix: ``A = L L^T``."""

    L: np.ndarray
    """ndarray, shape (n, n): Lower-triangular factor."""


@dataclass
class EigenResult:
    """Output of a symmetric eigenvalue algorithm."""

    eigenvalues: np.ndarray
    """ndarray, shape (n,) or scalar-like shape (1,): Eigenvalue(s) found."""

    eigenvectors: np.ndarray
    """ndarray, shape (n, n) or (n,): Corresponding eigenvector(s), as columns."""

    iterations: int = 0
    """int: Number of iterations/sweeps performed."""

    converged: bool = True
    """bool: Whether the convergence tolerance was met."""

    method: str = ""
    """str: e.g. ``"jacobi"``, ``"qr_algorithm"``, ``"power_iteration"``, ``"inverse_iteration"``."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics slot (e.g. Jacobi's off-diagonal norm history)."""


@dataclass
class SVDResult:
    """Output of a singular value decomposition: ``A = U diag(S) V^T``."""

    U: np.ndarray
    """ndarray, shape (m, k): Left singular vectors."""

    S: np.ndarray
    """ndarray, shape (k,): Singular values, descending."""

    Vt: np.ndarray
    """ndarray, shape (k, n): Right singular vectors, transposed."""


@dataclass
class IterativeSolveResult:
    """Output of an iterative linear-system solver (CG, GMRES)."""

    x: np.ndarray
    """ndarray, shape (n,): Approximate solution."""

    residual_history: np.ndarray
    """ndarray, shape (iterations + 1,): ``||b - A x_k||`` at each iterate."""

    iterations: int = 0
    """int: Number of iterations performed."""

    converged: bool = True
    """bool: Whether the residual tolerance was met."""

    method: str = ""
    """str: ``"conjugate_gradient"`` or ``"gmres"``."""


@dataclass
class LeastSquaresResult:
    """Output of a least-squares solve (normal equations or QR-based)."""

    coefficients: np.ndarray
    """ndarray, shape (n,): Least-squares solution."""

    residual_norm: float = 0.0
    """float: ``||b - A x||_2`` at the solution."""

    method: str = ""
    """str: ``"normal_equations"`` or ``"qr"``."""

    condition_number: Optional[float] = None
    """float, optional: 2-norm condition number of ``A`` (``"qr"``) or of
    the normal-equations matrix ``A^T A`` (``"normal_equations"``, which
    squares it)."""


class IterativeLinearSolver(ABC):
    """Common base for Krylov-subspace iterative linear-system solvers.

    Parameters
    ----------
    tol : float
        Relative residual-norm convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    """

    def __init__(self, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    @abstractmethod
    def solve(self, a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> IterativeSolveResult:
        """Solve ``A x = b`` iteratively.

        Parameters
        ----------
        a : ndarray, shape (n, n)
        b : ndarray, shape (n,)
        x0 : ndarray, shape (n,), optional
            Initial guess; defaults to the zero vector.

        Returns
        -------
        IterativeSolveResult
        """
