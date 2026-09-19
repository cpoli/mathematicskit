r"""The gamma and beta functions, via :mod:`scipy.special`.

``scipy.special.gamma``/``beta`` already implement these to full
``float64`` precision (via well-tested rational/asymptotic
approximations); mathematicskit does not reimplement them. See Abramowitz &
Stegun, *Handbook of Mathematical Functions*, Ch. 6, and NIST *Digital
Library of Mathematical Functions*, Ch. 5.
"""

from __future__ import annotations

from scipy import special

__all__ = ["gamma_function", "beta_function", "log_gamma_function"]


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
