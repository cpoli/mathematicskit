"""Polynomial evaluation and root sensitivity: Horner's scheme and
Wilkinson's ill-conditioned polynomial.

Horner's scheme is what :func:`numpy.polyval` runs internally; it is
written out in :func:`horner` because the by-product (the deflated
quotient and the derivative) is the point. See W. G. Horner, "A new
method of solving numerical equations of all orders, by continuous
approximation," Philosophical Transactions of the Royal Society of
London 109 (1819), 308-335; J. H. Wilkinson, "The evaluation of the
zeros of ill-conditioned polynomials. Part I," Numerische Mathematik 1
(1959), 150-166; Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 2.6.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.numerical_analysis.core.base import HornerResult

__all__ = ["horner", "wilkinson_polynomial", "root_condition_numbers"]


def horner(coefficients, x0: float) -> HornerResult:
    r"""Evaluate a polynomial and its derivative at ``x0`` by Horner's scheme.

    Writing :math:`p(x) = a_n x^n + \dots + a_1 x + a_0` in nested form
    :math:`(\cdots((a_n x + a_{n-1}) x + a_{n-2}) \cdots) x + a_0` needs
    only :math:`n` multiplications and :math:`n` additions, which is optimal
    (Ostrowski 1954, Pan 1966). The intermediate values
    :math:`b_k = b_{k+1} x_0 + a_k` are the coefficients of the quotient
    :math:`q` in :math:`p(x) = (x - x_0) q(x) + p(x_0)` (synthetic
    division), and a second pass gives :math:`p'(x_0) = q(x_0)`.

    Parameters
    ----------
    coefficients : array-like, shape (n + 1,)
        Coefficients, highest power first (``numpy.polyval`` convention).
    x0 : float
        Evaluation point.

    Returns
    -------
    HornerResult

    Examples
    --------
    >>> # p(x) = 2x^3 - 6x^2 + 2x - 1 at x = 3
    >>> result = horner([2.0, -6.0, 2.0, -1.0], 3.0)
    >>> result.value, result.derivative
    (5.0, 20.0)
    >>> result.quotient.tolist()  # 2x^2 + 0x + 2
    [2.0, 0.0, 2.0]
    """
    a = np.atleast_1d(np.asarray(coefficients, dtype=np.float64))
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError("coefficients must be a non-empty 1-D array")
    x0 = float(x0)
    b = np.empty_like(a)
    b[0] = a[0]
    for k in range(1, a.shape[0]):
        b[k] = b[k - 1] * x0 + a[k]
    quotient = b[:-1]
    derivative = 0.0
    for bk in quotient:
        derivative = derivative * x0 + bk
    return HornerResult(value=float(b[-1]), derivative=float(derivative), quotient=quotient.copy())


def wilkinson_polynomial(n: int = 20) -> np.ndarray:
    r"""Coefficients of Wilkinson's polynomial :math:`w(x) = \prod_{k=1}^{n} (x - k)`.

    Built with :func:`numpy.poly`. For ``n = 20`` several coefficients
    exceed :math:`2^{53}` and are already rounded in double precision, and
    the roots are so sensitive that changing the :math:`x^{19}` coefficient
    :math:`-210` by :math:`2^{-23}` sends ten of them into the complex
    plane (Wilkinson 1959, 1963).

    Parameters
    ----------
    n : int
        Degree (number of roots :math:`1, \ldots, n`).

    Returns
    -------
    ndarray, shape (n + 1,)
        Coefficients, highest power first.

    Examples
    --------
    >>> wilkinson_polynomial(3).tolist()  # (x-1)(x-2)(x-3)
    [1.0, -6.0, 11.0, -6.0]
    >>> float(wilkinson_polynomial(20)[1])
    -210.0
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    return np.poly(np.arange(1, n + 1, dtype=np.float64))


def root_condition_numbers(coefficients, root: float) -> np.ndarray:
    r"""Relative condition number of a simple root with respect to each coefficient.

    Perturbing :math:`a_k` to :math:`a_k(1 + \varepsilon)` moves a simple
    root :math:`r` by :math:`\delta r \approx -\varepsilon\,a_k r^k / p'(r)`,
    so the relative condition number is

    .. math::

        \kappa_k = \frac{|a_k|\,|r|^k}{|r|\,|p'(r)|}.

    See Wilkinson, *Rounding Errors in Algebraic Processes* (1963), Ch. 2;
    Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
    (SIAM, 2002), Ch. 26.

    Parameters
    ----------
    coefficients : array-like, shape (n + 1,)
        Coefficients, highest power first.
    root : float
        A simple, nonzero root of the polynomial.

    Returns
    -------
    ndarray, shape (n + 1,)
        :math:`\kappa_k`, in the same order as `coefficients`.

    Examples
    --------
    >>> # (x - 1)(x - 2) = x^2 - 3x + 2, root 2, p'(2) = 1
    >>> root_condition_numbers([1.0, -3.0, 2.0], 2.0).tolist()
    [2.0, 3.0, 1.0]
    """
    a = np.asarray(coefficients, dtype=np.float64)
    r = float(root)
    if r == 0.0:
        raise ValueError("root must be nonzero for a relative condition number")
    dp = horner(a, r).derivative
    if dp == 0.0:
        raise ValueError("root must be simple (p'(root) != 0)")
    powers = np.arange(a.shape[0] - 1, -1, -1)
    return np.abs(a) * np.abs(r) ** powers / (abs(r) * abs(dp))
