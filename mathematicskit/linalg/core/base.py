"""Abstract base classes and result containers for mathematicskit.linalg.

Every decomposition/solver in this domain returns a small dataclass
(mirroring physicskit's ``SimulationResult`` pattern). Most of this
domain's algorithms call ``numpy.linalg``/``scipy.linalg``/
``scipy.sparse.linalg`` directly (see each ``systems/`` module's
docstring for the specific library routine); the Krylov-subspace linear
solvers (:mod:`mathematicskit.linalg.systems.iterative`) share the
:class:`IterativeLinearSolver` ABC, the same shape as
:mod:`mathematicskit.numerical_analysis.core.base.IterativeRootFinder`, purely
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

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL

__all__ = [
    "LUResult",
    "QRResult",
    "CholeskyResult",
    "EigenResult",
    "SVDResult",
    "IterativeSolveResult",
    "LeastSquaresResult",
    "GershgorinResult",
    "SchurResult",
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
    """Output of an eigenvalue computation, whole-spectrum or single-pair.

    The whole-spectrum solvers
    (:func:`~mathematicskit.linalg.systems.eigen.eigen_symmetric`,
    :func:`~mathematicskit.linalg.systems.eigen.eigen_general`) fill
    `eigenvalues`/`eigenvectors` with the full spectrum in one shot and
    report ``iterations=1, converged=True``, since LAPACK does not expose
    its internal iteration count. The hand-rolled iterations
    (:func:`~mathematicskit.linalg.systems.eigen.power_iteration`,
    :func:`~mathematicskit.linalg.systems.eigen.inverse_iteration`) return a
    single eigenpair and a genuine iteration count and convergence flag.
    """

    eigenvalues: np.ndarray
    """ndarray: The whole spectrum, shape (n,), for the library-backed
    solvers; a single dominant (or shift-selected) eigenvalue, shape (1,),
    for power/inverse iteration. Real and ascending from ``eigh``, possibly
    complex and unordered from ``eig``."""

    eigenvectors: np.ndarray
    """ndarray: Shape (n, n) with eigenvectors as *columns* for the
    whole-spectrum solvers, so ``eigenvectors[:, k]`` pairs with
    ``eigenvalues[k]``; a single unit eigenvector of shape (n,) for
    power/inverse iteration."""

    iterations: int = 0
    """int: Number of iterations performed (always 1 for the library-backed
    solvers, which do not expose LAPACK's internal count)."""

    converged: bool = True
    """bool: Whether the convergence tolerance was met (always ``True`` for
    the direct library solvers, which do not iterate visibly)."""

    method: str = ""
    """str: One of ``"numpy_eigh"``, ``"numpy_eig"``, ``"power_iteration"``,
    or ``"inverse_iteration"``."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics slot, unused by the current solvers."""


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
    """ndarray, shape (iterations + 1,): The *relative* residual norm
    ``||b - A x_k|| / ||b||`` at each iterate, starting from the initial
    guess. Normalizing by ``||b||`` is what makes the history directly
    comparable to `method`'s ``tol`` (itself a relative tolerance) and
    comparable across right-hand sides of different magnitudes."""

    iterations: int = 0
    """int: Number of iterations performed."""

    converged: bool = True
    """bool: Whether the residual tolerance was met."""

    method: str = ""
    """str: ``"conjugate_gradient"``, ``"gmres"``, ``"jacobi"``,
    ``"gauss_seidel"``, or ``"sor"``."""


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


@dataclass
class GershgorinResult:
    r"""The Gershgorin discs of a square matrix.

    Disc :math:`k` is centered at :math:`a_{kk}` with radius
    :math:`R_k = \sum_{j \ne k} |a_{kj}|`; every eigenvalue lies in the
    union of the discs.
    """

    centers: np.ndarray
    """ndarray, shape (n,): Disc centers, the diagonal entries (possibly complex)."""

    radii: np.ndarray
    """ndarray, shape (n,): Disc radii, the off-diagonal absolute row (or column) sums."""

    def contains(self, z: complex) -> bool:
        """Whether the complex number `z` lies in the union of the discs (boundary included, up to rounding)."""
        return bool(np.any(np.abs(z - self.centers) <= self.radii * (1.0 + 1e-12) + 1e-12))


@dataclass
class SchurResult:
    """Output of a Schur decomposition: ``A = Z T Z^H``."""

    T: np.ndarray
    """ndarray, shape (n, n): Schur form -- upper triangular (complex
    output) or quasi-upper-triangular with 2x2 blocks for complex-conjugate
    eigenvalue pairs (real output). Its diagonal (or block) entries are the
    eigenvalues of ``A``."""

    Z: np.ndarray
    """ndarray, shape (n, n): Unitary (orthogonal, for real output) Schur vectors."""

    @property
    def eigenvalues(self) -> np.ndarray:
        """ndarray, shape (n,): Eigenvalues read off ``T`` (2x2 diagonal blocks resolved with :func:`numpy.linalg.eigvals`)."""
        return np.linalg.eigvals(self.T) if np.isrealobj(self.T) else np.diag(self.T).copy()


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
