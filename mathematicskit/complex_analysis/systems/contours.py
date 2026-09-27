r"""Contours, contour integrals, winding numbers, and Cauchy's integral formula.

:func:`contour_integral` evaluates :math:`\oint_\gamma f(z)\,dz =
\int_{t_0}^{t_1} f(\gamma(t))\,\gamma'(t)\,dt` piece by piece with
:func:`scipy.integrate.quad` (``complex_func=True``), splitting at the
contour's breakpoints so each call sees a smooth integrand. See
A.-L. Cauchy, *Mémoire sur les intégrales définies, prises entre des
limites imaginaires* (Paris, 1825), and L. V. Ahlfors, *Complex
Analysis*, 3rd ed. (McGraw-Hill, 1979), Ch. 4.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy import integrate

from mathematicskit.complex_analysis.core.base import Contour
from mathematicskit.constants import DEFAULT_ATOL, DEFAULT_RTOL

__all__ = ["circle_contour", "polygon_contour", "contour_integral", "winding_number", "cauchy_integral_formula"]


def circle_contour(center: complex = 0.0, radius: float = 1.0) -> Contour:
    r"""The counter-clockwise circle :math:`\gamma(t) = c + r e^{it}`, :math:`t \in [0, 2\pi]`.

    Parameters
    ----------
    center : complex
    radius : float
        Must be positive.

    Returns
    -------
    Contour

    Examples
    --------
    >>> c = circle_contour(1j, 2.0)
    >>> complex(np.round(c.gamma(0.0), 12))
    (2+1j)
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    return Contour(
        gamma=lambda t: center + radius * np.exp(1j * t),
        dgamma=lambda t: 1j * radius * np.exp(1j * t),
        breakpoints=np.array([0.0, 2.0 * np.pi]),
    )


def polygon_contour(vertices: Sequence[complex]) -> Contour:
    r"""The closed polygon through ``vertices`` in order, returning to the first.

    Side :math:`k` is parametrized as :math:`v_k + (t - k)(v_{k+1} - v_k)`
    for :math:`t \in [k, k+1]`. Listing the vertices counter-clockwise
    gives a positively oriented contour; clockwise gives winding
    number :math:`-1` about interior points.

    Parameters
    ----------
    vertices : sequence of complex
        At least three vertices; do not repeat the first at the end.

    Returns
    -------
    Contour

    Examples
    --------
    >>> square = polygon_contour([0, 1, 1 + 1j, 1j])
    >>> complex(square.gamma(2.5))
    (0.5+1j)
    """
    v = np.asarray(vertices, dtype=complex)
    if v.size < 3:
        raise ValueError("a polygon needs at least three vertices")
    closed = np.append(v, v[0])
    sides = np.diff(closed)
    n = v.size

    def _side(t):
        return np.clip(np.floor(t).astype(int), 0, n - 1)

    def gamma(t):
        t = np.asarray(t, dtype=float)
        k = _side(t)
        return closed[k] + (t - k) * sides[k]

    def dgamma(t):
        return sides[_side(np.asarray(t, dtype=float))]

    return Contour(gamma=gamma, dgamma=dgamma, breakpoints=np.arange(n + 1, dtype=float))


def contour_integral(f: Callable, contour: Contour, epsabs: float = DEFAULT_ATOL, epsrel: float = DEFAULT_RTOL, limit: int = 200) -> complex:
    r""":math:`\oint_\gamma f(z)\,dz = \int f(\gamma(t))\,\gamma'(t)\,dt`, via :func:`scipy.integrate.quad`.

    Parameters
    ----------
    f : callable
        ``f(z) -> complex``, continuous on the contour.
    contour : Contour
    epsabs, epsrel : float
        Tolerances passed to :func:`scipy.integrate.quad` for each smooth piece.
    limit : int
        Maximum number of adaptive subintervals per piece.

    Returns
    -------
    complex

    Examples
    --------
    >>> value = contour_integral(lambda z: 1 / z, circle_contour())
    >>> complex(np.round(value, 10)) == complex(0, round(2 * np.pi, 10))
    True
    """
    total = 0j
    # quad samples only interior points, so each call sees one smooth piece.
    for a, b in zip(contour.breakpoints[:-1], contour.breakpoints[1:], strict=False):
        value, _ = integrate.quad(
            lambda t: complex(f(contour.gamma(t)) * contour.dgamma(t)), a, b, complex_func=True, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        total += value
    return complex(total)


def winding_number(contour: Contour, z0: complex) -> int:
    r"""The winding number :math:`n(\gamma, z_0) = \frac{1}{2\pi i}\oint_\gamma \frac{dz}{z - z_0}`.

    Parameters
    ----------
    contour : Contour
        A closed contour not passing through ``z0``.
    z0 : complex

    Returns
    -------
    int
        The integral rounded to the nearest integer (it is an integer
        in exact arithmetic).

    Examples
    --------
    >>> winding_number(circle_contour(), 0.5j), winding_number(circle_contour(), 2.0)
    (1, 0)
    """
    value = contour_integral(lambda z: 1.0 / (z - z0), contour) / (2j * np.pi)
    return int(round(value.real))


def cauchy_integral_formula(f: Callable, contour: Contour, z0: complex, n: int = 0) -> complex:
    r"""Cauchy's integral formula :math:`f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_\gamma \frac{f(z)}{(z - z_0)^{n+1}}\,dz`.

    Recovers the value (``n = 0``) or any derivative of a holomorphic
    ``f`` at ``z0`` from its values on a contour winding once around
    ``z0``. See A.-L. Cauchy, "Sur la mécanique céleste et sur un
    nouveau calcul appelé calcul des limites" (Turin, 1831); Ahlfors,
    *Complex Analysis*, Ch. 4, Sec. 2.3.

    Parameters
    ----------
    f : callable
        Holomorphic inside and on ``contour``.
    contour : Contour
        Positively oriented, winding once around ``z0``.
    z0 : complex
    n : int
        Derivative order, ``n >= 0``.

    Returns
    -------
    complex

    Examples
    --------
    >>> value = cauchy_integral_formula(np.exp, circle_contour(), 0.3, n=2)
    >>> round(value.real, 10) == round(float(np.exp(0.3)), 10)
    True
    """
    if n < 0:
        raise ValueError(f"derivative order must be non-negative, got {n}")
    integral = contour_integral(lambda z: f(z) / (z - z0) ** (n + 1), contour)
    return complex(math.factorial(n) * integral / (2j * np.pi))
