"""Lanczos iteration for a few extreme eigenvalues of a large symmetric matrix.

A thin wrapper around :func:`scipy.sparse.linalg.eigsh`, which calls
ARPACK's implicitly restarted Lanczos method (Sorensen 1992). Only
matrix-vector products with ``A`` are needed (plus one sparse LU
factorization in the optional shift-invert mode), so ``A`` may be a
sparse matrix or a :class:`scipy.sparse.linalg.LinearOperator`, and a
handful of eigenpairs of a matrix far too large for :func:`numpy.linalg.eigh`
cost only a few hundred products. See C. Lanczos, J. Res. Nat. Bur.
Standards 45(4) (1950), 255-282, and Golub & Van Loan, *Matrix
Computations*, 4th ed., Ch. 10.1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse.linalg as sla

from mathematicskit.linalg.core.base import EigenResult

__all__ = ["lanczos_eigsh"]


def lanczos_eigsh(a, k: int = 6, which: str = "LA", tol: float = 0.0, sigma: Optional[float] = None) -> EigenResult:
    r"""The ``k`` extreme eigenpairs of a symmetric matrix, via :func:`scipy.sparse.linalg.eigsh` (Lanczos/ARPACK).

    Lanczos builds an orthonormal basis :math:`Q_m` of the Krylov
    subspace :math:`\operatorname{span}\{v, Av, \dots, A^{m-1}v\}` with
    a three-term recurrence, in which
    :math:`T_m = Q_m^T A Q_m` is tridiagonal; the eigenvalues of
    :math:`T_m` (Ritz values) converge first to the extreme eigenvalues
    of ``A``.

    Parameters
    ----------
    a : ndarray, sparse matrix, or LinearOperator, shape (n, n)
        Symmetric (Hermitian) matrix.
    k : int
        Number of eigenpairs, ``k < n``.
    which : {"LA", "SA", "LM", "SM", "BE"}
        Largest/smallest algebraic, largest/smallest magnitude, or both
        ends of the spectrum.
    tol : float
        Relative accuracy for the Ritz values; ``0`` means machine
        precision.
    sigma : float, optional
        Shift-invert mode: run Lanczos on :math:`(A - \sigma I)^{-1}`
        (factored once by sparse LU), which returns the ``k`` eigenvalues
        nearest ``sigma`` when ``which="LM"``. The standard route to
        interior or tightly clustered eigenvalues, which plain Lanczos
        resolves only slowly.

    Returns
    -------
    EigenResult
        ``eigenvalues`` shape (k,) ascending, ``eigenvectors`` shape
        (n, k), ``method="lanczos_arpack"``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.diag(np.arange(1.0, 21.0))
    >>> np.round(lanczos_eigsh(A, k=3).eigenvalues, 8)
    array([18., 19., 20.])
    >>> np.round(lanczos_eigsh(A, k=2, sigma=7.2, which="LM").eigenvalues, 8)
    array([7., 8.])
    """
    v0 = np.ones(a.shape[0])
    eigenvalues, eigenvectors = sla.eigsh(a, k=k, which=which, tol=tol, v0=v0, sigma=sigma)
    order = np.argsort(eigenvalues)
    return EigenResult(eigenvalues=eigenvalues[order], eigenvectors=eigenvectors[:, order], iterations=1, converged=True, method="lanczos_arpack")
