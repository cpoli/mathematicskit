r"""The error function and the Fresnel integrals, via :mod:`scipy.special`.

Both are integrals of a Gaussian: :math:`\operatorname{erf}` along the
real axis, and the Fresnel integrals :math:`C + iS` along the diagonal
:math:`t(1+i)` of the complex plane (Abramowitz & Stegun, Eq. 7.3.22).
See Abramowitz & Stegun, *Handbook of Mathematical Functions*, Ch. 7,
and NIST *Digital Library of Mathematical Functions*, Ch. 7.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from mathematicskit.special_functions.core.base import FresnelResult

__all__ = ["error_function", "complementary_error_function", "fresnel_integrals"]


def error_function(x):
    r"""The error function :math:`\operatorname{erf}(x) = \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}\,dt`.

    The probability that a normal variable lies within :math:`x\sqrt2`
    standard deviations of its mean. Named by J. W. L. Glaisher, "On a
    class of definite integrals," Philosophical Magazine 42 (1871),
    294-302. Via :func:`scipy.special.erf`.

    Parameters
    ----------
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> round(float(error_function(1 / math.sqrt(2))), 4)  # the "68%" of the 68-95-99.7 rule
    0.6827
    """
    return special.erf(x)


def complementary_error_function(x):
    r"""The complementary error function :math:`\operatorname{erfc}(x) = 1 - \operatorname{erf}(x)`.

    Via :func:`scipy.special.erfc`, which stays accurate in the far tail
    where computing ``1 - erf(x)`` directly would cancel to zero.

    Parameters
    ----------
    x : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> float(complementary_error_function(10.0)) > 0.0  # 1 - erf(10) is exactly 0.0 in float64
    True
    """
    return special.erfc(x)


def fresnel_integrals(t):
    r"""The Fresnel integrals :math:`S(t) = \int_0^t \sin(\pi\tau^2/2)\,d\tau` and :math:`C(t) = \int_0^t \cos(\pi\tau^2/2)\,d\tau`.

    The parametric curve :math:`(C(t), S(t))` is the Euler (Cornu)
    spiral, which winds into the points :math:`\pm(\tfrac12, \tfrac12)`
    as :math:`t\to\pm\infty`. Via :func:`scipy.special.fresnel`. See
    A. Fresnel, "Mémoire sur la diffraction de la lumière," Mémoires de
    l'Académie Royale des Sciences de l'Institut de France 5 (1826),
    339-475.

    Parameters
    ----------
    t : float or array-like of float

    Returns
    -------
    FresnelResult

    Examples
    --------
    >>> r = fresnel_integrals(1.0)
    >>> round(float(r.c), 6), round(float(r.s), 6)
    (0.779893, 0.438259)
    """
    t = np.asarray(t, dtype=float)
    s, c = special.fresnel(t)
    return FresnelResult(t=t, s=s, c=c)
