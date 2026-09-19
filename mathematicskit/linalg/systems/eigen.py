"""Eigenvalue computation (library), and power/inverse iteration (hand-rolled).

Eigenvalue computation itself is built directly on
:func:`numpy.linalg.eigh` (symmetric/Hermitian matrices, via LAPACK's
``?syevd``) and :func:`numpy.linalg.eig` (general matrices, via LAPACK's
``?geev``) -- there is no reason to hand-roll cyclic Jacobi or unshifted
QR iteration when these are already correct, well-tested, and much
faster. Power iteration and inverse iteration are kept hand-rolled: the
*iteration itself* -- watching a single eigenpair emerge from repeated
matrix-vector products -- is the pedagogical subject, and neither has a
direct library equivalent (LAPACK solves for the whole spectrum at once).
See Golub & Van Loan, *Matrix Computations*, 4th ed., Ch. 8.2-8.5, and
Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 9.3-9.5.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER
from mathematicskit.linalg.core.base import EigenResult

__all__ = ["eigen_symmetric", "eigen_general", "power_iteration", "inverse_iteration"]


def eigen_symmetric(a: np.ndarray) -> EigenResult:
    r"""Eigenvalues/eigenvectors of a symmetric (Hermitian) matrix, via :func:`numpy.linalg.eigh`.

    ``eigh`` exploits symmetry to guarantee real eigenvalues and an
    orthonormal eigenvector basis, returned in ascending order, at
    roughly half the cost of the general eigenvalue problem. See Golub &
    Van Loan, *Matrix Computations*, 4th ed., Ch. 8.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Symmetric matrix.

    Returns
    -------
    EigenResult
        ``eigenvalues`` shape (n,) ascending, ``eigenvectors`` shape
        (n, n) with columns the corresponding eigenvectors,
        ``method="numpy_eigh"``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0], [1.0, 2.0]])
    >>> result = eigen_symmetric(A)
    >>> np.round(result.eigenvalues, 8)
    array([1., 3.])
    """
    a = np.asarray(a, dtype=np.float64)
    if not np.allclose(a, a.T, atol=1e-10):
        raise ValueError("a must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    return EigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors, iterations=1, converged=True, method="numpy_eigh")


def eigen_general(a: np.ndarray) -> EigenResult:
    r"""Eigenvalues/eigenvectors of a general (possibly non-symmetric) matrix, via :func:`numpy.linalg.eig`.

    Eigenvalues/eigenvectors may be complex even for a real input matrix
    (e.g. a rotation matrix). See Golub & Van Loan, *Matrix Computations*,
    4th ed., Ch. 7.

    Parameters
    ----------
    a : ndarray, shape (n, n)

    Returns
    -------
    EigenResult
        ``method="numpy_eig"``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90-degree rotation
    >>> result = eigen_general(A)
    >>> np.round(np.sort(np.abs(result.eigenvalues.imag)), 8)
    array([1., 1.])
    """
    a = np.asarray(a, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(a)
    return EigenResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors, iterations=1, converged=True, method="numpy_eig")


def power_iteration(a: np.ndarray, tol: float = 1e-12, max_iter: int = DEFAULT_MAX_ITER, v0: Optional[np.ndarray] = None) -> EigenResult:
    r"""Power iteration for the dominant (largest-magnitude) eigenpair.

    :math:`v_{k+1} = A v_k / \|A v_k\|` converges to the eigenvector of
    the eigenvalue of largest magnitude (assuming it is unique and the
    initial vector has a nonzero component along it), with the Rayleigh
    quotient :math:`v_k^T A v_k` converging to the eigenvalue itself. No
    library equivalent for the iteration itself -- watching this converge
    is the point. See Burden & Faires, *Numerical Analysis*, 10th ed.,
    Ch. 9.3.

    Parameters
    ----------
    a : ndarray, shape (n, n)
    tol : float
        Convergence tolerance on the Rayleigh-quotient estimate.
    max_iter : int
    v0 : ndarray, shape (n,), optional
        Initial vector; a fixed pseudo-random default is used if omitted.

    Returns
    -------
    EigenResult
        ``eigenvalues`` a length-1 array, ``eigenvectors`` shape (n,).

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 0.0], [0.0, 5.0]])
    >>> result = power_iteration(A)
    >>> round(float(result.eigenvalues[0]), 6)
    5.0
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    v = np.random.default_rng(0).normal(size=n) if v0 is None else np.asarray(v0, dtype=np.float64)
    v = v / np.linalg.norm(v)
    lam_prev = 0.0
    converged = False
    it = 0
    while it < max_iter:
        it += 1
        w = a @ v
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-300:
            raise np.linalg.LinAlgError("iterate collapsed to zero")
        v = w / w_norm
        lam = float(v @ a @ v)
        if abs(lam - lam_prev) < tol * max(abs(lam), 1.0):
            converged = True
            break
        lam_prev = lam
    return EigenResult(eigenvalues=np.array([lam]), eigenvectors=v, iterations=it, converged=converged, method="power_iteration")


def inverse_iteration(a: np.ndarray, mu: float, tol: float = 1e-12, max_iter: int = DEFAULT_MAX_ITER, v0: Optional[np.ndarray] = None) -> EigenResult:
    r"""Inverse iteration for the eigenvalue nearest a shift ``mu``.

    Power iteration applied to :math:`(A - \mu I)^{-1}`: since that
    matrix's dominant eigenvalue is :math:`1/(\lambda - \mu)` for the
    :math:`\lambda` closest to ``mu``, iterating it converges to the
    corresponding eigenvector, with the Rayleigh quotient of the
    *original* ``A`` giving that eigenvalue directly. The linear solve at
    each step uses :func:`numpy.linalg.solve` rather than an explicit
    inverse; only the iteration itself is hand-rolled. See Burden &
    Faires, *Numerical Analysis*, 10th ed., Ch. 9.3, and Golub & Van
    Loan, *Matrix Computations*, 4th ed., Ch. 8.2.2.

    Parameters
    ----------
    a : ndarray, shape (n, n)
    mu : float
        Shift; should be close to the desired eigenvalue.
    tol : float
    max_iter : int
    v0 : ndarray, shape (n,), optional

    Returns
    -------
    EigenResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 0.0], [0.0, 5.0]])
    >>> result = inverse_iteration(A, mu=4.5)
    >>> round(float(result.eigenvalues[0]), 6)
    5.0
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    shifted = a - mu * np.eye(n)
    v = np.random.default_rng(0).normal(size=n) if v0 is None else np.asarray(v0, dtype=np.float64)
    v = v / np.linalg.norm(v)
    lam_prev = 0.0
    converged = False
    it = 0
    while it < max_iter:
        it += 1
        w = np.linalg.solve(shifted, v)
        w_norm = np.linalg.norm(w)
        v = w / w_norm
        lam = float(v @ a @ v)
        if abs(lam - lam_prev) < tol * max(abs(lam), 1.0):
            converged = True
            break
        lam_prev = lam
    return EigenResult(eigenvalues=np.array([lam]), eigenvectors=v, iterations=it, converged=converged, method="inverse_iteration")
