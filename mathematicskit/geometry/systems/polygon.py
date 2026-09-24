r"""Polygon area and centroid via the shoelace formula.

Simple enough that hand-writing it is arithmetic, not "reimplementing a
library" -- there's no meaningful sense in which
``scipy``/``numpy`` "already implements" the shoelace formula the way
they implement, say, a convex hull algorithm. See Bourke (1988), *Calculating
the area and centroid of a polygon*, and de Berg et al., *Computational
Geometry*, 3rd ed., Ch. 2 (polygon fundamentals).
"""

from __future__ import annotations

from math import gcd

import numpy as np

from mathematicskit.geometry.core.base import LatticePolygonResult
from mathematicskit.geometry.systems.intersections import point_in_polygon

__all__ = ["polygon_area", "polygon_centroid", "heron_area", "lattice_point_counts"]


def polygon_area(vertices: np.ndarray) -> float:
    r"""Signed area of a simple polygon via the shoelace formula.

    :math:`A = \tfrac12 \left|\sum_{i} (x_i y_{i+1} - x_{i+1} y_i)\right|`,
    indices modulo ``n`` (the polygon implicitly closes back to its
    first vertex). Positive for counterclockwise-ordered vertices,
    negative for clockwise (the absolute value is returned here).

    Parameters
    ----------
    vertices : ndarray, shape (n, 2)
        Polygon vertices in order (either winding direction).

    Returns
    -------
    float

    Examples
    --------
    >>> polygon_area(np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]))
    12.0
    """
    v = np.asarray(vertices, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    r"""Centroid (center of mass, assuming uniform density) of a simple polygon.

    :math:`C_x = \dfrac{1}{6A}\sum_i (x_i+x_{i+1})(x_iy_{i+1}-x_{i+1}y_i)`,
    similarly for :math:`C_y` -- the shoelace-formula-weighted average,
    *not* the plain average of the vertex coordinates (which is only
    correct for a regular polygon). See Bourke (1988).

    Parameters
    ----------
    vertices : ndarray, shape (n, 2)

    Returns
    -------
    ndarray, shape (2,)

    Examples
    --------
    >>> polygon_centroid(np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]))
    array([2. , 1.5])
    """
    v = np.asarray(vertices, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)
    cx = np.sum((x + x_next) * cross) / (6.0 * signed_area)
    cy = np.sum((y + y_next) * cross) / (6.0 * signed_area)
    return np.array([cx, cy])


def heron_area(a: float, b: float, c: float) -> float:
    r"""Area of a triangle from its three side lengths, by Heron's formula.

    With semi-perimeter :math:`s = (a+b+c)/2`,
    :math:`A = \sqrt{s(s-a)(s-b)(s-c)}`. Evaluated in William Kahan's
    numerically stable arrangement, which stays accurate for needle-like
    triangles where the textbook form loses digits: with
    :math:`a \ge b \ge c`,

    .. math::

       A = \tfrac14\sqrt{(a+(b+c))(c-(a-b))(c+(a-b))(a+(b-c))}.

    See W. Kahan, "Miscalculating Area and Angles of a Needle-like
    Triangle" (lecture notes, University of California, Berkeley, 2014).

    Parameters
    ----------
    a, b, c : float
        Side lengths satisfying the triangle inequality.

    Returns
    -------
    float

    Examples
    --------
    >>> heron_area(3.0, 4.0, 5.0)
    6.0
    """
    a, b, c = sorted((float(a), float(b), float(c)), reverse=True)
    if c - (a - b) < 0:
        raise ValueError("side lengths violate the triangle inequality")
    return 0.25 * float(np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))))


def lattice_point_counts(vertices) -> LatticePolygonResult:
    r"""Count the lattice points inside and on the boundary of a simple polygon with integer vertices.

    Boundary points are counted edge by edge with
    :math:`\gcd(|\Delta x|, |\Delta y|)`; interior points are counted
    directly by testing every lattice point in the bounding box with
    :func:`~mathematicskit.geometry.systems.intersections.point_in_polygon`.
    Pick's theorem says the area equals :math:`I + B/2 - 1`; the result
    reports both sides so the theorem can be checked.

    Parameters
    ----------
    vertices : array_like of int, shape (n, 2)

    Returns
    -------
    LatticePolygonResult

    Examples
    --------
    >>> result = lattice_point_counts([[0, 0], [4, 0], [4, 3], [0, 3]])
    >>> result.interior, result.boundary, result.area, result.pick_area
    (6, 14, 12.0, 12.0)
    """
    v = np.asarray(vertices, dtype=np.int64)
    edges = np.roll(v, -1, axis=0) - v
    on_boundary = set()  # each edge contributes gcd(|dx|, |dy|) lattice points
    for (x0, y0), (dx, dy) in zip(v.tolist(), edges.tolist()):
        g = gcd(abs(dx), abs(dy))
        for k in range(g):
            on_boundary.add((x0 + k * dx // g, y0 + k * dy // g))
    interior = 0
    polygon = v.astype(float)
    for x in range(int(v[:, 0].min()), int(v[:, 0].max()) + 1):
        for y in range(int(v[:, 1].min()), int(v[:, 1].max()) + 1):
            if (x, y) not in on_boundary and point_in_polygon(np.array([x, y], dtype=float), polygon):
                interior += 1
    return LatticePolygonResult(interior=interior, boundary=len(on_boundary), area=polygon_area(polygon))
