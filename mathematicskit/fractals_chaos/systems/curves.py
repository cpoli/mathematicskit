r"""Classical fractal curves and the measurements that exposed them:
the Weierstrass function, the Koch curve, the Hilbert curve, Richardson's
divider (ruler) length, and the similarity dimension.

Hand-rolled constructions (no numpy/scipy equivalent); the similarity
dimension's Moran equation is solved with :func:`scipy.optimize.brentq`.
See Falconer, *Fractal Geometry*, 3rd ed., Ch. 9 (self-similar sets) and
Ch. 11 (graphs of functions), and Sagan, *Space-Filling Curves*
(New York: Springer, 1994), Ch. 2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = ["weierstrass_function", "koch_curve", "koch_snowflake", "hilbert_curve", "divider_length", "similarity_dimension"]


def weierstrass_function(x, a: float = 0.5, b: float = 7.0, n_terms: int = 30) -> np.ndarray:
    r"""Partial sum of the Weierstrass function :math:`W(x) = \sum_{n \ge 0} a^n \cos(b^n \pi x)`.

    Karl Weierstrass showed in 1872 that for :math:`0 < a < 1`, odd
    integer :math:`b`, and :math:`ab > 1 + 3\pi/2` the sum is continuous
    everywhere but differentiable nowhere. G. H. Hardy (1916) extended
    this to all :math:`ab \ge 1`. Its graph has box-counting dimension
    :math:`2 + \log a / \log b`.

    Parameters
    ----------
    x : array_like
    a : float
        Amplitude ratio, :math:`0 < a < 1`.
    b : float
        Frequency ratio.
    n_terms : int
        Number of terms kept.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> round(float(weierstrass_function(0.0, a=0.5, n_terms=40)), 6)  # sum of 0.5^n
    2.0
    """
    x = np.asarray(x, dtype=float)
    n = np.arange(n_terms)
    return np.sum(a ** n[:, None] * np.cos(np.pi * b ** n[:, None] * x.ravel()), axis=0).reshape(x.shape)


def koch_curve(order: int, start=(0.0, 0.0), end=(1.0, 0.0)) -> np.ndarray:
    r"""Vertices of the Koch curve after ``order`` refinements of a segment.

    Each step replaces every segment by four segments one third as long,
    with the middle third raised into an equilateral bump. Helge von
    Koch's 1904 construction is continuous, nowhere differentiable, and
    has similarity dimension :math:`\log 4/\log 3 \approx 1.2619`.

    Parameters
    ----------
    order : int
    start, end : tuple of float
        Endpoints of the initial segment.

    Returns
    -------
    ndarray, shape (4**order + 1, 2)

    Examples
    --------
    >>> koch_curve(2).shape
    (17, 2)
    """
    points = np.array([start, end], dtype=float)
    rotation = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    for _ in range(order):
        a, b = points[:-1], points[1:]
        step = (b - a) / 3
        p1, p3 = a + step, a + 2 * step
        p2 = p1 + step @ rotation.T
        refined = np.stack([a, p1, p2, p3], axis=1).reshape(-1, 2)
        points = np.vstack([refined, points[-1:]])
    return points


def koch_snowflake(order: int) -> np.ndarray:
    r"""Vertices of the closed Koch snowflake: three Koch curves on an equilateral triangle.

    Parameters
    ----------
    order : int

    Returns
    -------
    ndarray, shape (3 * 4**order + 1, 2)
        The first vertex is repeated at the end to close the curve.

    Examples
    --------
    >>> koch_snowflake(1).shape
    (13, 2)
    """
    corners = [(0.0, 0.0), (0.5, np.sqrt(3) / 2), (1.0, 0.0)]
    sides = [koch_curve(order, corners[i], corners[(i + 1) % 3])[:-1] for i in range(3)]
    closed = np.vstack(sides)
    return np.vstack([closed, closed[:1]])


def hilbert_curve(order: int) -> np.ndarray:
    r"""Vertices of the order-``order`` Hilbert curve, visiting every cell of a :math:`2^n \times 2^n` grid.

    David Hilbert's 1891 space-filling curve is the limit of these
    polygons. Consecutive vertices are grid neighbours, so points close
    along the curve stay close in the plane. Uses the standard
    index-to-coordinate conversion (Hacker's Delight, Sec. 16-2).

    Parameters
    ----------
    order : int

    Returns
    -------
    ndarray, shape (4**order, 2), int
        Integer grid coordinates in ``[0, 2**order)``.

    Examples
    --------
    >>> hilbert_curve(1).tolist()
    [[0, 0], [0, 1], [1, 1], [1, 0]]
    """
    n = 2**order
    points = np.zeros((n * n, 2), dtype=int)
    for index in range(n * n):
        x = y = 0
        t = index
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        points[index] = (x, y)
    return points


def divider_length(curve: np.ndarray, ruler: float) -> float:
    r"""Length of a polygonal curve measured by walking a pair of dividers of opening ``ruler`` along it.

    Starting at the first vertex, each step jumps to the first later
    point of the curve at distance ``ruler``. Lewis Fry Richardson found
    empirically that measured coastline lengths grow as
    :math:`L(\varepsilon) \propto \varepsilon^{1-D}` as the ruler
    :math:`\varepsilon` shrinks, and Benoit Mandelbrot identified
    :math:`D` as a fractal dimension.

    Parameters
    ----------
    curve : ndarray, shape (n, 2)
        Vertices of a polyline, finely sampled compared with ``ruler``.
    ruler : float

    Returns
    -------
    float
        Number of full steps times ``ruler``, plus the final partial step.

    Examples
    --------
    >>> line = np.column_stack([np.linspace(0, 1, 1001), np.zeros(1001)])
    >>> round(divider_length(line, 0.1), 10)
    1.0
    """
    curve = np.asarray(curve, dtype=float)
    position = curve[0]
    i = 1  # index of the first vertex beyond `position`
    steps = 0
    while True:
        distances = np.linalg.norm(curve[i:] - position, axis=1)
        beyond = np.nonzero(distances >= ruler * (1 - 1e-12))[0]  # tolerate rounding at exact hits
        if beyond.size == 0:
            return steps * ruler + float(np.linalg.norm(curve[-1] - position))
        j = i + beyond[0]
        # The point on segment curve[j-1] -> curve[j] at distance `ruler` from `position`.
        p, d = curve[j - 1], curve[j] - curve[j - 1]
        f = p - position
        qa, qb, qc = d @ d, 2 * f @ d, f @ f - ruler**2
        t = (-qb + np.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)
        position = p + min(max(t, 0.0), 1.0) * d
        i = j
        steps += 1


def similarity_dimension(ratios) -> float:
    r"""The similarity dimension :math:`D` solving Moran's equation :math:`\sum_i r_i^D = 1`.

    For a self-similar set built from maps with contraction ratios
    :math:`r_i` that do not overlap too much (the open set condition),
    Patrick Moran proved in 1946 that :math:`D` equals the Hausdorff
    dimension. With :math:`m` equal ratios :math:`r` it reduces to
    :math:`\log m / \log(1/r)`.

    Parameters
    ----------
    ratios : sequence of float
        Contraction ratios, each in :math:`(0, 1)`.

    Returns
    -------
    float

    Examples
    --------
    >>> round(similarity_dimension([1 / 3] * 4), 6)  # Koch curve: log 4 / log 3
    1.26186
    >>> round(similarity_dimension([0.5, 0.25, 0.25]), 6)  # 0.5^D + 2 * 0.25^D = 1
    1.0
    """
    ratios = np.asarray(ratios, dtype=float)
    if np.any((ratios <= 0) | (ratios >= 1)):
        raise ValueError("contraction ratios must lie in (0, 1)")
    return float(brentq(lambda d: np.sum(ratios**d) - 1.0, 0.0, 64.0))
