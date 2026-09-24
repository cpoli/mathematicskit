r"""The gamma and beta functions, via :mod:`scipy.special`.

``scipy.special.gamma``/``beta`` already implement these to full
``float64`` precision (via well-tested rational/asymptotic
approximations); mathematicskit does not reimplement them. See Abramowitz &
Stegun, *Handbook of Mathematical Functions*, Ch. 6, and NIST *Digital
Library of Mathematical Functions*, Ch. 5.

The one hand-written routine here is Stirling's asymptotic series for
:math:`\ln\Gamma(x)`, :func:`stirling_log_gamma`: scipy has no public
"truncated Stirling series" function, and watching the truncation error
shrink (then grow again) with the number of terms is the point.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["gamma_function", "beta_function", "log_gamma_function", "stirling_factorial", "stirling_log_gamma"]


def gamma_function(x):
    r"""The gamma function :math:`\Gamma(x) = \int_0^\infty t^{x-1}e^{-t}\,dt`.

    Extends the factorial to real (and complex) arguments:
    :math:`\Gamma(n) = (n-1)!` for positive integers ``n``. Via
    :func:`scipy.special.gamma`. See Abramowitz & Stegun, *Handbook of
    Mathematical Functions*, Sec. 6.1.

    Parameters
    ----------
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(float(gamma_function(5.0)), 6)  # 4! = 24
    24.0
    >>> round(float(gamma_function(0.5)), 6)  # sqrt(pi)
    1.772454
    """
    return special.gamma(x)


def log_gamma_function(x):
    r"""The (natural) log of the gamma function, :math:`\ln|\Gamma(x)|`.

    Via :func:`scipy.special.gammaln` -- numerically stable for large
    ``x``, where :math:`\Gamma(x)` itself overflows ``float64`` long
    before its logarithm does. See NIST *Digital Library of Mathematical
    Functions*, Sec. 5.4.

    Parameters
    ----------
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(float(log_gamma_function(171.0)), 4)
    706.5731
    """
    return special.gammaln(x)


def beta_function(a, b):
    r"""The beta function :math:`B(a,b) = \dfrac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)} = \int_0^1 t^{a-1}(1-t)^{b-1}\,dt`.

    Via :func:`scipy.special.beta`. See Abramowitz & Stegun, *Handbook of
    Mathematical Functions*, Sec. 6.2.

    Parameters
    ----------
    a, b : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> from scipy import special
    >>> round(float(beta_function(2.0, 3.0)), 6) == round(float(special.gamma(2.0) * special.gamma(3.0) / special.gamma(5.0)), 6)
    True
    """
    return special.beta(a, b)


def stirling_factorial(n):
    r"""Stirling's approximation :math:`n! \approx \sqrt{2\pi n}\,(n/e)^n`.

    The leading term of Stirling's series (J. Stirling, *Methodus
    Differentialis*, 1730, Prop. 28; A. de Moivre, *Miscellanea
    Analytica*, 1730). The *relative* error is about :math:`1/(12n)`, so
    the ratio to :math:`n!` tends to 1 even though the absolute error
    grows without bound.

    Parameters
    ----------
    n : float or array-like of float
        Non-negative argument.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> round(float(stirling_factorial(10.0)) / math.factorial(10), 4)
    0.9917
    """
    n = np.asarray(n, dtype=float)
    return np.sqrt(2.0 * np.pi * n) * (n / np.e) ** n


def stirling_log_gamma(x, terms=3):
    r"""Stirling's asymptotic series for :math:`\ln\Gamma(x)`, truncated after ``terms`` corrections.

    .. math::

       \ln\Gamma(x) \sim \left(x - \tfrac12\right)\ln x - x + \tfrac12\ln(2\pi)
       + \sum_{k=1}^{K} \frac{B_{2k}}{2k(2k-1)\,x^{2k-1}},

    where :math:`B_{2k}` are the Bernoulli numbers (from
    :func:`scipy.special.bernoulli`). The series is *asymptotic*, not
    convergent: for fixed :math:`x`, adding terms helps only up to
    roughly :math:`K \approx \pi x`, after which the error grows. See
    NIST *Digital Library of Mathematical Functions*, Eq. 5.11.1.

    Parameters
    ----------
    x : float or array-like of float
        Positive argument.
    terms : int, optional
        Number of Bernoulli correction terms :math:`K \ge 0`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> abs(float(stirling_log_gamma(10.0, terms=3)) - float(log_gamma_function(10.0))) < 1e-9
    True
    """
    x = np.asarray(x, dtype=float)
    result = (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi)
    if terms > 0:
        bernoulli = special.bernoulli(2 * terms)
        for k in range(1, terms + 1):
            result = result + bernoulli[2 * k] / (2 * k * (2 * k - 1) * x ** (2 * k - 1))
    return result
