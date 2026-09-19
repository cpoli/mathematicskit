"""Singular value decomposition, via :func:`numpy.linalg.svd`.

Production SVD implementations bidiagonalize ``A`` first (Golub-Kahan)
rather than forming :math:`A^T A` and eigendecomposing it -- squaring the
matrix squares its condition number, which is exactly the instability
:mod:`mathkit.linalg.systems.stability` demonstrates for the normal
equations. :func:`numpy.linalg.svd` (LAPACK's ``?gesdd``) already uses
the numerically stable route, so mathkit wraps it directly rather than
reimplementing the less-stable textbook definition. See Trefethen & Bau,
*Numerical Linear Algebra*, 1997, Lecture 4-5, and Golub & Van Loan,
*Matrix Computations*, 4th ed., Ch. 8.6.
"""

from __future__ import annotations

import numpy as np

from mathkit.linalg.core.base import SVDResult

__all__ = ["svd_decompose"]


def svd_decompose(a: np.ndarray) -> SVDResult:
    r"""Compute the (thin) SVD ``A = U diag(S) V^T`` via :func:`numpy.linalg.svd`.

    ``full_matrices=False`` requests the thin/economy factorization
    (``U`` shape (m, k), ``Vt`` shape (k, n) with ``k = min(m, n)``),
    which is what every :mod:`mathkit.linalg` caller needs. Singular
    values are returned in descending order (numpy's convention).

    Parameters
    ----------
    a : ndarray, shape (m, n)

    Returns
    -------
    SVDResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3.0, 0.0], [0.0, -2.0]])
    >>> result = svd_decompose(A)
    >>> np.round(result.S, 6)
    array([3., 2.])
    >>> np.allclose(result.U @ np.diag(result.S) @ result.Vt, A)
    True
    """
    a = np.asarray(a, dtype=np.float64)
    U, S, Vt = np.linalg.svd(a, full_matrices=False)
    return SVDResult(U=U, S=S, Vt=Vt)
