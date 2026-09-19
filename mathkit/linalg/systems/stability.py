"""Condition-number estimation and least-squares stability comparison:
normal equations vs. QR.

Condition number is a thin wrapper around :func:`numpy.linalg.cond`; the
"normal equations vs. QR" *comparison* is mathkit's value-add (scipy has
no single function that stages this stability lesson), but each solve
itself is built on library primitives -- :func:`numpy.linalg.solve` for
the normal equations and :func:`scipy.linalg.qr` +
:func:`scipy.linalg.solve_triangular` for the QR route. See Trefethen &
Bau, *Numerical Linear Algebra*, 1997, Lecture 12 (least squares) and
Lecture 19 (backward stability/conditioning), and Golub & Van Loan,
*Matrix Computations*, 4th ed., Ch. 5.3.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from mathkit.linalg.core.base import LeastSquaresResult

__all__ = ["condition_number_2norm", "least_squares_normal_equations", "least_squares_qr", "least_squares_lstsq"]


def condition_number_2norm(a: np.ndarray) -> float:
    r"""2-norm condition number :math:`\kappa_2(A) = \sigma_{\max}/\sigma_{\min}`, via :func:`numpy.linalg.cond`.

    A large :math:`\kappa_2` means small relative perturbations in ``A``
    or ``b`` can produce large relative changes in the solution of
    ``A x = b`` -- see :func:`least_squares_normal_equations` for why
    this matters concretely for least squares. See Trefethen & Bau,
    *Numerical Linear Algebra*, 1997, Lecture 12.

    Parameters
    ----------
    a : ndarray, shape (m, n)

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> round(condition_number_2norm(np.diag([10.0, 1.0])), 6)
    10.0
    """
    a = np.asarray(a, dtype=np.float64)
    cond = np.linalg.cond(a, p=2)
    if not np.isfinite(cond):
        raise np.linalg.LinAlgError("a is singular (or the zero matrix)")
    return float(cond)


def least_squares_normal_equations(a: np.ndarray, b: np.ndarray) -> LeastSquaresResult:
    r"""Least squares via the normal equations :math:`A^T A x = A^T b`.

    ``A^T A`` and ``A^T b`` are formed explicitly and solved with
    :func:`numpy.linalg.solve`. Simple, but
    :math:`\kappa_2(A^T A) = \kappa_2(A)^2` -- squaring the condition
    number of ``A`` itself -- so this can lose roughly twice as many
    digits of accuracy as the QR-based approach (:func:`least_squares_qr`)
    for an ill-conditioned ``A``. See Trefethen & Bau, *Numerical Linear
    Algebra*, 1997, Lecture 11.

    Parameters
    ----------
    a : ndarray, shape (m, n)
    b : ndarray, shape (m,)

    Returns
    -------
    LeastSquaresResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    >>> b = np.array([2.0, 3.0, 5.0])
    >>> result = least_squares_normal_equations(A, b)
    >>> np.round(result.coefficients, 4)
    array([0.3333, 1.5   ])
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ata = a.T @ a
    atb = a.T @ b
    x = np.linalg.solve(ata, atb)
    residual_norm = float(np.linalg.norm(b - a @ x))
    try:
        cond = condition_number_2norm(ata)
    except np.linalg.LinAlgError:
        cond = None
    return LeastSquaresResult(coefficients=x, residual_norm=residual_norm, method="normal_equations", condition_number=cond)


def least_squares_qr(a: np.ndarray, b: np.ndarray) -> LeastSquaresResult:
    r"""Least squares via ``A = Q R`` (:func:`scipy.linalg.qr`): solve ``R x = Q^T b``.

    Avoids ever forming :math:`A^T A`, so the effective conditioning is
    :math:`\kappa_2(A)` rather than its square -- the standard remedy for
    :func:`least_squares_normal_equations`'s instability. (For very
    ill-conditioned or rank-deficient problems, :func:`numpy.linalg.lstsq`
    -- SVD-based -- is even more robust; the QR route is used here
    specifically to make the comparison with the normal equations
    concrete.) See Trefethen & Bau, *Numerical Linear Algebra*, 1997,
    Lecture 11.

    Parameters
    ----------
    a : ndarray, shape (m, n), ``m >= n``
    b : ndarray, shape (m,)

    Returns
    -------
    LeastSquaresResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    >>> b = np.array([2.0, 3.0, 5.0])
    >>> result = least_squares_qr(A, b)
    >>> np.round(result.coefficients, 4)
    array([0.3333, 1.5   ])
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    q, r = sla.qr(a, mode="economic")
    qtb = q.T @ b
    x = sla.solve_triangular(r, qtb, lower=False)
    residual_norm = float(np.linalg.norm(b - a @ x))
    try:
        cond = condition_number_2norm(a)
    except np.linalg.LinAlgError:
        cond = None
    return LeastSquaresResult(coefficients=x, residual_norm=residual_norm, method="qr", condition_number=cond)


def least_squares_lstsq(a: np.ndarray, b: np.ndarray) -> LeastSquaresResult:
    r"""Least squares via :func:`numpy.linalg.lstsq` (SVD-based, via LAPACK ``?gelsd``).

    The most robust of the three routes here -- it handles rank-deficient
    ``a`` gracefully (returning the minimum-norm solution) where both
    :func:`least_squares_normal_equations` and :func:`least_squares_qr`
    would fail or become ill-conditioned -- because it never needs ``a``
    to have full column rank in the first place. See Trefethen & Bau,
    *Numerical Linear Algebra*, 1997, Lecture 11.

    Parameters
    ----------
    a : ndarray, shape (m, n)
    b : ndarray, shape (m,)

    Returns
    -------
    LeastSquaresResult
        ``method="lstsq"``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    >>> b = np.array([2.0, 3.0, 5.0])
    >>> result = least_squares_lstsq(A, b)
    >>> np.round(result.coefficients, 4)
    array([0.3333, 1.5   ])
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x, _residuals, _rank, _s = np.linalg.lstsq(a, b, rcond=None)
    residual_norm = float(np.linalg.norm(b - a @ x))
    try:
        cond = condition_number_2norm(a)
    except np.linalg.LinAlgError:
        cond = None
    return LeastSquaresResult(coefficients=x, residual_norm=residual_norm, method="lstsq", condition_number=cond)
