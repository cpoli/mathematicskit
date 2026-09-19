r"""Bessel functions of the first and second kind, via :mod:`scipy.special`.

``scipy.special.jv``/``yv`` implement these via well-tested series/
asymptotic expansions for any real order; mathematicskit does not reimplement
them. See Abramowitz & Stegun, *Handbook of Mathematical Functions*,
Ch. 9, and NIST *Digital Library of Mathematical Functions*, Ch. 10.
"""

from __future__ import annotations

from scipy import special

__all__ = ["bessel_first_kind", "bessel_second_kind"]


def bessel_first_kind(nu, x):
    r"""Bessel function of the first kind, :math:`J_\nu(x)`.

    Solves Bessel's differential equation :math:`x^2y'' + xy' + (x^2 -
    \nu^2)y = 0`, regular (finite) at :math:`x=0`. Via
    :func:`scipy.special.jv`. See Abramowitz & Stegun, *Handbook of
    Mathematical Functions*, Sec. 9.1.

    Parameters
    ----------
    nu : float
        Order.
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(float(bessel_first_kind(0.0, 0.0)), 6)  # J_0(0) = 1
    1.0
    >>> round(float(bessel_first_kind(1.0, 0.0)), 6)  # J_nu(0) = 0 for nu > 0
    0.0
    """
    return special.jv(nu, x)


def bessel_second_kind(nu, x):
    r"""Bessel function of the second kind (Weber/Neumann function), :math:`Y_\nu(x)`.

    The second, linearly independent solution of Bessel's equation,
    singular (diverging to :math:`-\infty`) at :math:`x=0`. Via
    :func:`scipy.special.yv`. See Abramowitz & Stegun, *Handbook of
    Mathematical Functions*, Sec. 9.1.

    Parameters
    ----------
    nu : float
        Order.
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> bool(bessel_second_kind(0.0, 0.1) < bessel_second_kind(0.0, 1.0))  # diverges toward x=0
    True
    """
    return special.yv(nu, x)
