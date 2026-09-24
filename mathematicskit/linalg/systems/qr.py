"""QR decomposition via Householder reflections (library) and via
Gram-Schmidt (hand-rolled, kept for the stability comparison).

Both factor ``A = Q R`` with ``Q`` orthonormal-columns and ``R`` upper
triangular, but differ sharply in numerical stability: Householder
reflections produce a ``Q`` that is orthogonal to machine precision
regardless of conditioning, while *classical* Gram-Schmidt can lose
orthogonality badly for ill-conditioned ``A`` -- the textbook motivating
example for why Householder is the default in practice (LAPACK's
``?geqrf``, wrapped here by :func:`scipy.linalg.qr`). The comparison
itself, not the Householder QR result, is why a from-scratch Gram-Schmidt
implementation is kept alongside the library call. See Trefethen & Bau,
*Numerical Linear Algebra*, 1997, Lectures 7-10, and Golub & Van Loan,
*Matrix Computations*, 4th ed., Ch. 5.1-5.2.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mathematicskit.linalg.core.base import QRResult

__all__ = ["householder_qr", "gram_schmidt_qr", "orthogonality_error"]


def householder_qr(a: np.ndarray) -> QRResult:
    r"""QR decomposition by Householder reflections, via :func:`scipy.linalg.qr`.

    Each column below the diagonal is zeroed by an orthogonal reflector
    :math:`H_k = I - 2 v_k v_k^T / (v_k^T v_k)` chosen to map that
    column's trailing part onto a multiple of :math:`e_1`; the product of
    all the :math:`H_k` is ``Q``. ``mode="economic"`` requests the thin
    factorization (``Q`` shape (m, n) rather than (m, m)) to match
    :func:`gram_schmidt_qr`'s shape. See Trefethen & Bau, *Numerical
    Linear Algebra*, 1997, Lecture 10.

    Parameters
    ----------
    a : ndarray, shape (m, n), ``m >= n``
        Full-column-rank matrix.

    Returns
    -------
    QRResult
        With ``method="householder"``, ``Q`` shape (m, n), ``R`` shape (n, n).

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 1.0]])
    >>> result = householder_qr(A)
    >>> np.allclose(result.Q @ result.R, A)
    True
    >>> np.allclose(result.Q.T @ result.Q, np.eye(2), atol=1e-10)
    True
    """
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    if m < n:
        raise ValueError("a must have at least as many rows as columns")
    q, r = sla.qr(a, mode="economic")
    return QRResult(Q=q, R=r, method="householder")


def gram_schmidt_qr(a: np.ndarray, modified: bool = True) -> QRResult:
    r"""QR decomposition by (modified or classical) Gram-Schmidt.

    Orthogonalizes ``A``'s columns one at a time against the
    previously-computed ``Q`` columns. *Modified* Gram-Schmidt
    (``modified=True``, the default) subtracts each projection
    immediately as it's computed rather than all at once from the
    original vector (*classical* Gram-Schmidt, ``modified=False``) --
    algebraically equivalent in exact arithmetic, but modified
    Gram-Schmidt loses far less orthogonality in floating point,
    especially for near-collinear columns. See Trefethen & Bau,
    *Numerical Linear Algebra*, 1997, Lecture 8, and
    :func:`orthogonality_error` for a direct numerical comparison.

    Parameters
    ----------
    a : ndarray, shape (m, n), ``m >= n``
        Full-column-rank matrix.
    modified : bool
        Whether to use modified (default) or classical Gram-Schmidt.

    Returns
    -------
    QRResult
        With ``method="gram_schmidt"``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 1.0]])
    >>> result = gram_schmidt_qr(A)
    >>> np.allclose(result.Q @ result.R, A)
    True
    """
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    if m < n:
        raise ValueError("a must have at least as many rows as columns")
    q = np.zeros((m, n))
    r = np.zeros((n, n))

    if modified:
        v = a.copy()
        for k in range(n):
            r[k, k] = np.linalg.norm(v[:, k])
            if r[k, k] < 1e-300:
                raise np.linalg.LinAlgError("columns of a are linearly dependent")
            q[:, k] = v[:, k] / r[k, k]
            for j in range(k + 1, n):
                r[k, j] = q[:, k] @ v[:, j]
                v[:, j] -= r[k, j] * q[:, k]
    else:
        for k in range(n):
            v = a[:, k].copy()
            for j in range(k):
                r[j, k] = q[:, j] @ a[:, k]
                v -= r[j, k] * q[:, j]
            r[k, k] = np.linalg.norm(v)
            if r[k, k] < 1e-300:
                raise np.linalg.LinAlgError("columns of a are linearly dependent")
            q[:, k] = v / r[k, k]

    return QRResult(Q=q, R=r, method="gram_schmidt" if modified else "gram_schmidt_classical")


def orthogonality_error(q: np.ndarray) -> float:
    r"""Measure how far ``Q``'s columns are from exactly orthonormal.

    :math:`\max_{ij}\left|(Q^TQ - I)_{ij}\right|`: the largest entry of
    ``Q``'s Gram matrix minus the identity. (This is the entrywise max
    norm, not the induced :math:`\infty`-norm, which would be the largest
    row *sum*.) Exactly zero for a perfectly orthonormal ``Q``, and a
    direct numerical demonstration of which QR method loses orthogonality
    on ill-conditioned input: on a degree-12 Vandermonde matrix, classical
    Gram-Schmidt reaches order 1, modified Gram-Schmidt stays near 1e-10,
    and Householder near machine epsilon. See Trefethen & Bau, *Numerical
    Linear Algebra*, 1997, Lecture 9 (Figure 9.2's classical-vs-modified
    comparison).

    Parameters
    ----------
    q : ndarray, shape (m, n)

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> orthogonality_error(np.eye(3)) == 0.0
    True
    """
    n = q.shape[1]
    return float(np.max(np.abs(q.T @ q - np.eye(n))))
