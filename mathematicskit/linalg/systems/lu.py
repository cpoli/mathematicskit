"""LU decomposition with partial pivoting, and the solve built on it.

Built directly on :func:`scipy.linalg.lu` for the factorization and
:func:`scipy.linalg.solve_triangular` for the forward/back-substitution
solves -- Gaussian elimination with partial pivoting is a well-understood,
already-correct LAPACK routine (``dgetrf``/``?getrf``), so mathematicskit does not
reimplement it; the value here is the package's own :class:`LUResult`
dataclass, permutation-matrix bookkeeping, and determinant/solve
convenience wrappers around it. See Burden & Faires, *Numerical Analysis*,
10th ed., Ch. 6.4-6.5, and Golub & Van Loan, *Matrix Computations*, 4th
ed., Ch. 3.2-3.4, for the underlying algorithm this LAPACK routine
implements.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mathematicskit.linalg.core.base import LUResult

__all__ = ["lu_decompose", "lu_solve", "lu_det", "lu_solve_system"]


def lu_decompose(a: np.ndarray) -> LUResult:
    r"""Factor ``P A = L U`` by Gaussian elimination with partial pivoting.

    Thin wrapper around :func:`scipy.linalg.lu`, which brings the
    largest-magnitude entry in each column to the pivot position by a row
    swap before eliminating below it -- the standard remedy for the
    numerical instability of naive (unpivoted) Gaussian elimination.
    :func:`scipy.linalg.lu` returns a permutation matrix ``P0`` with
    ``A = P0 L U``; this function returns its transpose so that, matching
    the textbook convention, ``result.P @ a == result.L @ result.U``. See
    Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 6.5 ("Matrix
    Factorizations"), and Trefethen & Bau, *Numerical Linear Algebra*,
    1997, Lecture 21.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Square, nonsingular matrix.

    Returns
    -------
    LUResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]])
    >>> result = lu_decompose(A)
    >>> np.allclose(result.P @ A, result.L @ result.U)
    True
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("a must be square")
    if abs(sla.det(a)) < 1e-300:
        raise np.linalg.LinAlgError("matrix is singular to working precision")

    p0, L, U = sla.lu(a)
    P = p0.T
    # Parity of the row permutation encoded by P: the minimum number of
    # transpositions to realize it is n minus its number of cycles.
    perm = np.argmax(P, axis=1)
    visited = np.zeros(n, dtype=bool)
    num_cycles = 0
    for i in range(n):
        if not visited[i]:
            num_cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
    num_swaps = n - num_cycles

    return LUResult(L=L, U=U, P=P, num_row_swaps=num_swaps)


def lu_solve(result: LUResult, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` given a precomputed :class:`LUResult` for ``A``.

    Solves ``L y = P b`` then ``U x = y`` by triangular substitution
    (:func:`scipy.linalg.solve_triangular`, :math:`O(n^2)`) rather than
    refactoring at :math:`O(n^3)` -- the whole point of reusing a
    factorization across several right-hand sides.

    Parameters
    ----------
    result : LUResult
        From :func:`lu_decompose`.
    b : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([3.0, 5.0])
    >>> x = lu_solve(lu_decompose(A), b)
    >>> np.allclose(A @ x, b)
    True
    """
    pb = result.P @ np.asarray(b, dtype=np.float64)
    y = sla.solve_triangular(result.L, pb, lower=True, unit_diagonal=True)
    x = sla.solve_triangular(result.U, y, lower=False)
    return x


def lu_solve_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience: factor ``a`` and solve in one call.

    Uses :func:`scipy.linalg.lu_factor`/:func:`scipy.linalg.lu_solve`
    directly (LAPACK's ``?getrf``/``?getrs`` pair) rather than routing
    through :func:`lu_decompose`'s explicit ``P``/``L``/``U``, which is
    marginally faster since it skips reconstructing the dense factors.

    Parameters
    ----------
    a : ndarray, shape (n, n)
    b : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]])
    >>> x_true = np.array([1.0, -2.0, -2.0])
    >>> b = A @ x_true
    >>> np.allclose(lu_solve_system(A, b), x_true, atol=1e-8)
    True
    """
    lu_and_piv = sla.lu_factor(np.asarray(a, dtype=np.float64))
    return sla.lu_solve(lu_and_piv, np.asarray(b, dtype=np.float64))


def lu_det(a: np.ndarray) -> float:
    r"""Determinant via LU decomposition: :math:`\det A = (-1)^s \prod_i U_{ii}`.

    Thin wrapper around :func:`scipy.linalg.det`, which itself factors
    ``a`` via LAPACK's ``?getrf`` and takes the signed product of ``U``'s
    diagonal -- :math:`O(n^3)`, versus the :math:`O(n!)` cofactor
    expansion. See Burden & Faires, *Numerical Analysis*, 10th ed., Ch.
    6.5.

    Parameters
    ----------
    a : ndarray, shape (n, n)

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> round(lu_det(A), 10)
    -2.0
    """
    return float(sla.det(np.asarray(a, dtype=np.float64)))
