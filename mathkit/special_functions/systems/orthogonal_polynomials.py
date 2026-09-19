r"""Orthogonal polynomial families: Legendre, Chebyshev, Hermite, Laguerre.

Built directly on :mod:`numpy.polynomial`'s per-family submodules, which
already implement stable evaluation (Clenshaw recurrence) and every
family's three-term recurrence; mathkit does not reimplement them. See
Abramowitz & Stegun, *Handbook of Mathematical Functions*, Ch. 22, and
Arfken, Weber & Harris, *Mathematical Methods for Physicists*, 7th ed.,
Ch. 18.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev, hermite, laguerre, legendre

__all__ = ["legendre_polynomial", "chebyshev_polynomial", "hermite_polynomial", "laguerre_polynomial"]


def legendre_polynomial(n: int, x):
    r"""The Legendre polynomial :math:`P_n(x)`, orthogonal on :math:`[-1,1]` with weight 1.

    Via :func:`numpy.polynomial.legendre.legval` on the basis
    coefficients for degree ``n``. Solutions of Legendre's differential
    equation; arise e.g. in the multipole expansion of the
    electrostatic/gravitational potential. See Arfken, Weber & Harris,
    *Mathematical Methods for Physicists*, 7th ed., Ch. 18.2.

    Parameters
    ----------
    n : int
    x : float or array-like of float, in ``[-1, 1]``

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import numpy as np
    >>> round(float(legendre_polynomial(2, 1.0)), 6)  # P_n(1) = 1 for all n
    1.0
    >>> np.round(legendre_polynomial(2, np.array([-1.0, 0.0, 1.0])), 6)
    array([ 1. , -0.5,  1. ])
    """
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return legendre.legval(x, coeffs)


def chebyshev_polynomial(n: int, x):
    r"""The Chebyshev polynomial (first kind) :math:`T_n(x) = \cos(n\arccos x)`, orthogonal on :math:`[-1,1]` with weight :math:`1/\sqrt{1-x^2}`.

    Via :func:`numpy.polynomial.chebyshev.chebval`. The minimal-sup-norm
    property of (monic-scaled) Chebyshev polynomials is exactly why
    Chebyshev *nodes* (see :mod:`mathkit.numerical_analysis.systems.chebyshev`)
    avoid the Runge phenomenon. See Arfken, Weber & Harris, *Mathematical
    Methods for Physicists*, 7th ed., Ch. 18.4.

    Parameters
    ----------
    n : int
    x : float or array-like of float, in ``[-1, 1]``

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(float(chebyshev_polynomial(3, 0.5)), 6)  # T_3(cos(pi/3)) = cos(pi) = -1
    -1.0
    """
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return chebyshev.chebval(x, coeffs)


def hermite_polynomial(n: int, x):
    r"""The (physicists') Hermite polynomial :math:`H_n(x)`, orthogonal on :math:`(-\infty,\infty)` with weight :math:`e^{-x^2}`.

    Via :func:`numpy.polynomial.hermite.hermval`. Arise as the
    eigenfunctions of the quantum harmonic oscillator. See Arfken, Weber
    & Harris, *Mathematical Methods for Physicists*, 7th ed., Ch. 18.3.

    Parameters
    ----------
    n : int
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> float(hermite_polynomial(0, 2.0))
    1.0
    >>> float(hermite_polynomial(1, 2.0))  # H_1(x) = 2x
    4.0
    >>> round(float(hermite_polynomial(2, 1.0)), 6)  # H_2(x) = 4x^2 - 2
    2.0
    """
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return hermite.hermval(x, coeffs)


def laguerre_polynomial(n: int, x):
    r"""The (simple) Laguerre polynomial :math:`L_n(x)`, orthogonal on :math:`[0,\infty)` with weight :math:`e^{-x}`.

    Via :func:`numpy.polynomial.laguerre.lagval`. Arise as the radial
    part of the hydrogen atom's wavefunctions. See Arfken, Weber &
    Harris, *Mathematical Methods for Physicists*, 7th ed., Ch. 18.5.

    Parameters
    ----------
    n : int
    x : float or array-like of float, ``x >= 0``

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> float(laguerre_polynomial(0, 3.0))
    1.0
    >>> round(float(laguerre_polynomial(1, 1.0)), 6)  # L_1(x) = 1 - x
    0.0
    """
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return laguerre.lagval(x, coeffs)
