r"""Elliptic integrals, the arithmetic-geometric mean, and Jacobi elliptic functions, via :mod:`scipy.special`.

Uses the *parameter* convention :math:`m = k^2` throughout, matching
:func:`scipy.special.ellipk`/:func:`~scipy.special.ellipe`/
:func:`~scipy.special.ellipj` and Abramowitz & Stegun, *Handbook of
Mathematical Functions*, Ch. 16-17. Gauss's 1799 discovery that
:math:`K(m) = \pi / (2\,\mathrm{AGM}(1, \sqrt{1-m}))` is checked in
this domain's tests against :func:`scipy.special.agm`.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from mathematicskit.special_functions.core.base import JacobiEllipticResult

__all__ = [
    "arithmetic_geometric_mean",
    "complete_elliptic_integral_first_kind",
    "complete_elliptic_integral_second_kind",
    "jacobi_elliptic_functions",
]


def arithmetic_geometric_mean(a, b):
    r"""Gauss's arithmetic-geometric mean :math:`\mathrm{AGM}(a, b)`.

    Iterate :math:`a_{n+1} = (a_n + b_n)/2`, :math:`b_{n+1} =
    \sqrt{a_n b_n}`; both sequences converge quadratically to a common
    limit. Via :func:`scipy.special.agm`. See C. F. Gauss, *Werke*,
    vol. 3 (1866), 361-403.

    Parameters
    ----------
    a, b : float or array-like of float
        Non-negative arguments.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> # Gauss's constant 1/AGM(1, sqrt 2) = 0.8346268...
    >>> round(1 / float(arithmetic_geometric_mean(1.0, math.sqrt(2.0))), 7)
    0.8346268
    """
    return special.agm(a, b)


def complete_elliptic_integral_first_kind(m):
    r"""The complete elliptic integral of the first kind, :math:`K(m) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - m\sin^2\theta}}`.

    Via :func:`scipy.special.ellipk` (parameter :math:`m = k^2`). See
    Abramowitz & Stegun, Sec. 17.3.

    Parameters
    ----------
    m : float or array-like of float
        Parameter, :math:`m < 1`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> abs(float(complete_elliptic_integral_first_kind(0.0)) - math.pi / 2) < 1e-15
    True
    """
    return special.ellipk(m)


def complete_elliptic_integral_second_kind(m):
    r"""The complete elliptic integral of the second kind, :math:`E(m) = \int_0^{\pi/2} \sqrt{1 - m\sin^2\theta}\,d\theta`.

    :math:`4aE(e^2)` is the perimeter of an ellipse with semi-major axis
    :math:`a` and eccentricity :math:`e`, the problem that gave elliptic
    integrals their name. Via :func:`scipy.special.ellipe`. See
    Abramowitz & Stegun, Sec. 17.3.

    Parameters
    ----------
    m : float or array-like of float
        Parameter, :math:`m \le 1`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> float(complete_elliptic_integral_second_kind(1.0))  # degenerate ellipse: a segment
    1.0
    """
    return special.ellipe(m)


def jacobi_elliptic_functions(u, m):
    r"""The Jacobi elliptic functions :math:`\operatorname{sn}`, :math:`\operatorname{cn}`, :math:`\operatorname{dn}`.

    Defined by inverting the incomplete elliptic integral of the first
    kind: if :math:`u = \int_0^\varphi d\theta/\sqrt{1 - m\sin^2\theta}`
    then :math:`\operatorname{sn} u = \sin\varphi`,
    :math:`\operatorname{cn} u = \cos\varphi`,
    :math:`\operatorname{dn} u = \sqrt{1 - m\sin^2\varphi}`. They are
    doubly periodic; on the real axis, :math:`\operatorname{sn}` and
    :math:`\operatorname{cn}` have period :math:`4K(m)`. Via
    :func:`scipy.special.ellipj`. See C. G. J. Jacobi, *Fundamenta nova
    theoriae functionum ellipticarum* (1829).

    Parameters
    ----------
    u : float or array-like of float
    m : float
        Parameter, :math:`0 \le m \le 1`.

    Returns
    -------
    JacobiEllipticResult

    Examples
    --------
    >>> r = jacobi_elliptic_functions(0.7, 0.5)
    >>> round(float(r.sn**2 + r.cn**2), 12)
    1.0
    """
    sn, cn, dn, ph = special.ellipj(np.asarray(u, dtype=float), m)
    return JacobiEllipticResult(sn=sn, cn=cn, dn=dn, amplitude=ph)
