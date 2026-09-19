"""Cholesky decomposition for symmetric positive-definite matrices.

Built directly on :func:`numpy.linalg.cholesky` for the factorization and
:func:`scipy.linalg.solve_triangular` for the two triangular solves --
LAPACK's ``?potrf``/``?potrs`` are already the correct, well-tested
implementation of Burden & Faires, *Numerical Analysis*, 10th ed., Ch.
6.6 ("Special Types of Matrices"), Theorem 6.25 and Algorithm "Cholesky",
and Golub & Van Loan, *Matrix Computations*, 4th ed., Ch. 4.2.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mathkit.linalg.core.base import CholeskyResult

__all__ = ["cholesky_decompose", "cholesky_solve", "is_symmetric_positive_definite"]


def cholesky_decompose(a: np.ndarray) -> CholeskyResult:
    r"""Factor a symmetric positive-definite ``A = L L^T`` via :func:`numpy.linalg.cholesky`.

    Roughly half the cost of a general LU factorization (exploits
    symmetry); the underlying LAPACK routine (``?potrf``) raises when a
    diagonal pivot would require a square root of a negative number,
    which doubles as a certificate that ``a`` is not positive-definite.
    See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 6.6, Theorem
    6.25.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Symmetric positive-definite matrix.

    Returns
    -------
    CholeskyResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> result = cholesky_decompose(A)
    >>> np.allclose(result.L @ result.L.T, A)
    True
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be square")
    if not np.allclose(a, a.T, atol=1e-10):
        raise ValueError("a must be symmetric")
    try:
        L = np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("a is not positive definite") from exc
    return CholeskyResult(L=L)


def cholesky_solve(result: CholeskyResult, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` given a precomputed :class:`CholeskyResult`.

    Forward substitution with ``L``, then back substitution with ``L^T``,
    via :func:`scipy.linalg.solve_triangular`.

    Parameters
    ----------
    result : CholeskyResult
    b : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> b = np.array([2.0, 1.0])
    >>> x = cholesky_solve(cholesky_decompose(A), b)
    >>> np.allclose(A @ x, b)
    True
    """
    L = result.L
    b = np.asarray(b, dtype=np.float64)
    y = sla.solve_triangular(L, b, lower=True)
    x = sla.solve_triangular(L.T, y, lower=False)
    return x


def is_symmetric_positive_definite(a: np.ndarray) -> bool:
    """Check symmetric positive-definiteness by attempting a Cholesky factorization.

    Parameters
    ----------
    a : ndarray, shape (n, n)

    Returns
    -------
    bool

    Examples
    --------
    >>> import numpy as np
    >>> is_symmetric_positive_definite(np.array([[4.0, 2.0], [2.0, 5.0]]))
    True
    >>> is_symmetric_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
    False
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T, atol=1e-10):
        return False
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False
