r"""Taylor/Maclaurin series expansion and convergence-radius estimation
for standard functions.

See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 1.1 (Taylor's
Theorem) and any standard calculus text (e.g. Stewart, *Calculus*, 8th
ed., Ch. 11.10) for the specific Maclaurin series below; the
ratio-test convergence radius is the standard result that a power series
:math:`\sum a_n x^n` converges for :math:`|x| < R`,
:math:`R = \lim_{n\to\infty} |a_n / a_{n+1}|` (when the limit exists).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["maclaurin_coefficients", "evaluate_series", "estimate_radius_of_convergence", "taylor_remainder_bound"]


def maclaurin_coefficients(name: str, order: int) -> np.ndarray:
    r"""Maclaurin-series coefficients :math:`a_0, \dots, a_{\text{order}}` for a standard function.

    Supported ``name`` values and their series:

    - ``"exp"``: :math:`e^x = \sum_{n\geq0} x^n/n!`, :math:`R=\infty`.
    - ``"sin"``: :math:`\sin x = \sum_{n\geq0} (-1)^n x^{2n+1}/(2n+1)!`, :math:`R=\infty`.
    - ``"cos"``: :math:`\cos x = \sum_{n\geq0} (-1)^n x^{2n}/(2n)!`, :math:`R=\infty`.
    - ``"log1p"``: :math:`\ln(1+x) = \sum_{n\geq1} (-1)^{n+1} x^n/n`, :math:`R=1`.
    - ``"geometric"``: :math:`1/(1-x) = \sum_{n\geq0} x^n`, :math:`R=1`.
    - ``"arctan"``: :math:`\arctan x = \sum_{n\geq0} (-1)^n x^{2n+1}/(2n+1)`, :math:`R=1`.

    Parameters
    ----------
    name : str
        One of the supported function names above.
    order : int
        Highest power to include (``order + 1`` coefficients returned).

    Returns
    -------
    ndarray, shape (order + 1,)
        ``a[n]`` is the coefficient of :math:`x^n`.

    Examples
    --------
    >>> import numpy as np
    >>> coeffs = maclaurin_coefficients("exp", 4)
    >>> np.allclose(coeffs, [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0])
    True
    """
    n = np.arange(order + 1)
    if name == "exp":
        return np.array([1.0 / math.factorial(k) for k in n])
    if name == "sin":
        a = np.zeros(order + 1)
        for k in n:
            if k % 2 == 1:
                a[k] = (-1.0) ** ((k - 1) // 2) / math.factorial(k)
        return a
    if name == "cos":
        a = np.zeros(order + 1)
        for k in n:
            if k % 2 == 0:
                a[k] = (-1.0) ** (k // 2) / math.factorial(k)
        return a
    if name == "log1p":
        a = np.zeros(order + 1)
        for k in n:
            if k >= 1:
                a[k] = (-1.0) ** (k + 1) / k
        return a
    if name == "geometric":
        return np.ones(order + 1)
    if name == "arctan":
        a = np.zeros(order + 1)
        for k in n:
            if k % 2 == 1:
                a[k] = (-1.0) ** ((k - 1) // 2) / k
        return a
    raise ValueError(f"unknown series name {name!r}")


def evaluate_series(coefficients: np.ndarray, x) -> float:
    r"""Evaluate a truncated power series :math:`\sum_n a_n x^n` (Horner's method).

    Parameters
    ----------
    coefficients : ndarray, shape (order + 1,)
        ``a[n]`` is the coefficient of :math:`x^n` (as from
        :func:`maclaurin_coefficients`).
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import numpy as np
    >>> coeffs = maclaurin_coefficients("exp", 15)
    >>> round(float(evaluate_series(coeffs, 1.0)), 10) == round(np.e, 10)
    True
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x) + coefficients[-1]
    for a in coefficients[-2::-1]:
        result = result * x + a
    return result


def estimate_radius_of_convergence(coefficients: np.ndarray) -> float:
    r"""Estimate a power series' radius of convergence via the ratio test.

    :math:`R \approx \lim_{n\to\infty} |a_n / a_{n+1}|`, estimated here
    from the last two nonzero coefficients (appropriate for series like
    ``"sin"``/``"cos"`` whose odd or even coefficients are exactly zero).
    See any standard calculus text's treatment of the ratio test for
    power series (e.g. Stewart, *Calculus*, 8th ed., Ch. 11.8).

    Parameters
    ----------
    coefficients : ndarray, shape (order + 1,)

    Returns
    -------
    float
        ``inf`` if the series appears to have unbounded radius (fewer
        than 2 nonzero coefficients to compare, or a genuinely growing
        ratio).

    Examples
    --------
    >>> coeffs = maclaurin_coefficients("geometric", 30)
    >>> round(estimate_radius_of_convergence(coeffs), 6)
    1.0
    """
    nonzero_idx = np.flatnonzero(coefficients)
    if nonzero_idx.shape[0] < 2:
        return float("inf")
    i, j = nonzero_idx[-2], nonzero_idx[-1]
    a_i, a_j = coefficients[i], coefficients[j]
    if a_j == 0.0:
        return float("inf")
    # Ratio test between consecutive *nonzero* terms, adjusted for the
    # power gap (j - i) between them (relevant for sin/cos, whose
    # nonzero terms are every other coefficient).
    ratio = abs(a_i / a_j) ** (1.0 / (j - i))
    return float(ratio)


def taylor_remainder_bound(max_derivative_bound: float, order: int, x: float, x0: float = 0.0) -> float:
    r"""Lagrange remainder bound for a degree-``order`` Taylor polynomial.

    :math:`|R_n(x)| \leq \dfrac{M}{(n+1)!}|x-x_0|^{n+1}`, where ``M``
    bounds :math:`|f^{(n+1)}|` on the interval between ``x0`` and ``x``.
    See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 1.1,
    Theorem 1.14 (Taylor's Theorem).

    Parameters
    ----------
    max_derivative_bound : float
        A bound ``M`` on :math:`|f^{(\text{order}+1)}|` on the relevant interval.
    order : int
        Degree of the Taylor polynomial.
    x, x0 : float
        Evaluation point and expansion center.

    Returns
    -------
    float

    Examples
    --------
    >>> # sin has |f^(n+1)| <= 1 everywhere, so this bounds the 5th-order
    >>> # Maclaurin remainder at x=0.5.
    >>> round(taylor_remainder_bound(1.0, 5, 0.5), 8)
    2.17e-05
    """
    return abs(max_derivative_bound) / math.factorial(order + 1) * abs(x - x0) ** (order + 1)
