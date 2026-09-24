r"""Mathieu functions, via :mod:`scipy.special`.

Mathieu's equation :math:`y'' + (a - 2q\cos 2x)\,y = 0` has
:math:`\pi`- or :math:`2\pi`-periodic solutions only for special
*characteristic values* :math:`a = a_m(q)` (even solutions
:math:`\operatorname{ce}_m`) or :math:`a = b_m(q)` (odd solutions
:math:`\operatorname{se}_m`). ``scipy.special`` takes angles in
**degrees**; these wrappers take radians. See É. Mathieu, "Mémoire sur
le mouvement vibratoire d'une membrane de forme elliptique," Journal de
Mathématiques Pures et Appliquées 13 (1868), 137-203, and NIST *Digital
Library of Mathematical Functions*, Ch. 28.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["mathieu_characteristic_a", "mathieu_characteristic_b", "mathieu_even", "mathieu_odd"]


def mathieu_characteristic_a(m, q):
    r"""Characteristic value :math:`a_m(q)` for the even Mathieu function :math:`\operatorname{ce}_m`.

    At :math:`q = 0` the equation is :math:`y'' + ay = 0` and
    :math:`a_m(0) = m^2`. Via :func:`scipy.special.mathieu_a`.

    Parameters
    ----------
    m : int
        Order, :math:`m \ge 0`.
    q : float

    Returns
    -------
    float

    Examples
    --------
    >>> float(mathieu_characteristic_a(3, 0.0))
    9.0
    """
    return special.mathieu_a(m, q)


def mathieu_characteristic_b(m, q):
    r"""Characteristic value :math:`b_m(q)` for the odd Mathieu function :math:`\operatorname{se}_m`.

    :math:`b_m(0) = m^2`. Via :func:`scipy.special.mathieu_b`.

    Parameters
    ----------
    m : int
        Order, :math:`m \ge 1`.
    q : float

    Returns
    -------
    float

    Examples
    --------
    >>> float(mathieu_characteristic_b(2, 0.0))
    4.0
    """
    return special.mathieu_b(m, q)


def mathieu_even(m, q, x):
    r"""The even Mathieu function :math:`\operatorname{ce}_m(x, q)`, with ``x`` in radians.

    Normalized so that :math:`\operatorname{ce}_m(x, 0) = \cos mx` for
    :math:`m \ge 1` (and :math:`1/\sqrt2` for :math:`m = 0`). Via
    :func:`scipy.special.mathieu_cem`.

    Parameters
    ----------
    m : int
    q : float
    x : float or array-like of float
        Angle in radians.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> round(float(mathieu_even(2, 0.0, 0.3)), 12) == round(math.cos(0.6), 12)
    True
    """
    value, _ = special.mathieu_cem(m, q, np.degrees(x))
    return value


def mathieu_odd(m, q, x):
    r"""The odd Mathieu function :math:`\operatorname{se}_m(x, q)`, with ``x`` in radians.

    Normalized so that :math:`\operatorname{se}_m(x, 0) = \sin mx`. Via
    :func:`scipy.special.mathieu_sem`.

    Parameters
    ----------
    m : int
        Order, :math:`m \ge 1`.
    q : float
    x : float or array-like of float
        Angle in radians.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> round(float(mathieu_odd(1, 0.0, 0.3)), 12) == round(math.sin(0.3), 12)
    True
    """
    value, _ = special.mathieu_sem(m, q, np.degrees(x))
    return value
