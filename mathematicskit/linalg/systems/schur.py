"""Hessenberg reduction and the Schur decomposition -- the output of the Francis QR algorithm.

Thin wrappers around :func:`scipy.linalg.hessenberg` (LAPACK ``?gehrd``)
and :func:`scipy.linalg.schur` (LAPACK ``?gees``, which runs the
implicitly shifted Francis QR algorithm on the Hessenberg form). The
Schur form ``A = Z T Z^H`` with ``T`` (quasi-)upper triangular exists
for every square matrix (Schur 1909) and exposes the eigenvalues on the
diagonal of ``T``. See J. G. F. Francis, *The Computer Journal* 4
(1961-1962), 265-271 and 332-345, and Golub & Van Loan, *Matrix
Computations*, 4th ed., Ch. 7.4-7.5.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mathematicskit.linalg.core.base import SchurResult

__all__ = ["hessenberg_reduce", "schur_decompose"]


def hessenberg_reduce(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Upper-Hessenberg form ``A = Q H Q^T`` via :func:`scipy.linalg.hessenberg`.

    ``H`` is zero below its first subdiagonal. Reducing to it first (by
    Householder reflections) cuts each QR step from :math:`O(n^3)` to
    :math:`O(n^2)`, and the QR iteration preserves the Hessenberg shape.

    Parameters
    ----------
    a : ndarray, shape (n, n)

    Returns
    -------
    H : ndarray, shape (n, n)
    Q : ndarray, shape (n, n)
        Orthogonal (unitary) similarity.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.arange(16.0).reshape(4, 4) + np.eye(4)
    >>> H, Q = hessenberg_reduce(A)
    >>> bool(np.allclose(np.tril(H, -2), 0.0)), bool(np.allclose(Q @ H @ Q.T, A))
    (True, True)
    """
    h, q = sla.hessenberg(np.asarray(a), calc_q=True)
    return h, q


def schur_decompose(a: np.ndarray, output: str = "real") -> SchurResult:
    r"""Schur decomposition ``A = Z T Z^H`` via :func:`scipy.linalg.schur`.

    Parameters
    ----------
    a : ndarray, shape (n, n)
    output : {"real", "complex"}
        ``"real"`` keeps a real ``T`` with 2x2 diagonal blocks for
        complex-conjugate eigenvalue pairs; ``"complex"`` returns a truly
        upper-triangular ``T``.

    Returns
    -------
    SchurResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [2.0, 3.0]])  # eigenvalues 2 and 5
    >>> result = schur_decompose(A)
    >>> np.round(np.sort(np.diag(result.T)), 8)
    array([2., 5.])
    >>> bool(np.allclose(result.Z @ result.T @ result.Z.T, A))
    True
    """
    t, z = sla.schur(np.asarray(a), output=output)
    return SchurResult(T=t, Z=z)
