"""Characteristic polynomials, matrix polynomials, and the Cayley-Hamilton theorem.

The characteristic polynomial's coefficients come from :func:`numpy.poly`
(which forms them from the eigenvalues computed by LAPACK); evaluating a
polynomial at a *matrix* argument has no numpy routine, so
:func:`matrix_polynomial` applies Horner's rule with matrix products.
The Cayley-Hamilton theorem states :math:`p_A(A) = 0` for every square
``A``. See A. Cayley, "A Memoir on the Theory of Matrices," Phil. Trans.
R. Soc. Lond. 148 (1858), 17-37, and Horn & Johnson, *Matrix Analysis*,
2nd ed., 2013, Ch. 2.4.
"""

from __future__ import annotations

import numpy as np

__all__ = ["characteristic_polynomial", "matrix_polynomial"]


def characteristic_polynomial(a: np.ndarray) -> np.ndarray:
    r"""Coefficients of :math:`p_A(\lambda) = \det(\lambda I - A)`, highest degree first, via :func:`numpy.poly`.

    The polynomial is monic, the coefficient of :math:`\lambda^{n-1}` is
    :math:`-\operatorname{tr} A`, and the constant term is
    :math:`(-1)^n \det A`.

    Parameters
    ----------
    a : ndarray, shape (n, n)

    Returns
    -------
    ndarray, shape (n + 1,)
        Real if ``a`` is real (imaginary round-off from complex-conjugate
        eigenvalue pairs is discarded).

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> np.round(characteristic_polynomial(A), 8)  # l^2 - 5 l - 2
    array([ 1., -5., -2.])
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be square")
    coeffs = np.poly(a)
    if np.isrealobj(a):
        coeffs = np.real(coeffs)
    return coeffs


def matrix_polynomial(coeffs: np.ndarray, a: np.ndarray) -> np.ndarray:
    r"""Evaluate :math:`p(A) = c_0 A^d + c_1 A^{d-1} + \dots + c_d I` by Horner's rule.

    With ``coeffs = characteristic_polynomial(a)`` the result is the zero
    matrix, up to rounding -- the Cayley-Hamilton theorem.

    Parameters
    ----------
    coeffs : array_like, shape (d + 1,)
        Polynomial coefficients, highest degree first (the
        :func:`numpy.polyval` convention).
    a : ndarray, shape (n, n)

    Returns
    -------
    ndarray, shape (n, n)

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> np.allclose(matrix_polynomial(characteristic_polynomial(A), A), 0.0)
    True
    """
    a = np.asarray(a)
    coeffs = np.atleast_1d(np.asarray(coeffs))
    n = a.shape[0]
    result = np.zeros((n, n), dtype=np.result_type(a, coeffs, np.float64))
    eye = np.eye(n)
    for c in coeffs:
        result = result @ a + c * eye
    return result
