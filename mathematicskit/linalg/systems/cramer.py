"""Cramer's rule: solving a square linear system by determinants.

Each determinant is computed with :func:`numpy.linalg.slogdet` (an LU
factorization via LAPACK ``?getrf``, returned as sign and log-magnitude
so that large ``n`` cannot overflow), so the rule costs :math:`n + 1`
factorizations -- :math:`O(n^4)` overall against elimination's
:math:`O(n^3)`. It is kept here for its historical and theoretical
value (an explicit formula for each unknown), not as a practical solver;
use :func:`~mathematicskit.linalg.systems.lu.lu_solve_system` for that.
See G. Cramer, *Introduction à l'analyse des lignes courbes algébriques*
(1750), Appendix I, and Meyer, *Matrix Analysis and Applied Linear
Algebra*, 2000, Ch. 6.2.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cramer_solve"]


def cramer_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Solve ``A x = b`` by Cramer's rule, :math:`x_i = \det(A_i) / \det(A)`.

    :math:`A_i` is ``A`` with its ``i``-th column replaced by ``b``.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Nonsingular coefficient matrix.
    b : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (n,)

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``det(A)`` is (numerically) zero.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([3.0, 5.0])
    >>> np.round(cramer_solve(A, b), 8)
    array([0.8, 1.4])
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    if a.shape != (n, n) or b.shape != (n,):
        raise ValueError("a must be square (n, n) and b of shape (n,)")
    # some BLAS builds (e.g. Apple Accelerate) raise spurious floating-point
    # flags inside ?getrf for larger n; the returned values are correct
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sign_a, logdet_a = np.linalg.slogdet(a)
        if sign_a == 0 or np.linalg.cond(a) > 1.0 / np.finfo(np.float64).eps:
            raise np.linalg.LinAlgError("a is singular: det(A) = 0")
        x = np.empty(n)
        for i in range(n):
            a_i = a.copy()
            a_i[:, i] = b
            # det(A_i) / det(A) via log-determinants, so large n cannot overflow
            sign_i, logdet_i = np.linalg.slogdet(a_i)
            x[i] = sign_i * sign_a * np.exp(logdet_i - logdet_a)
    return x
