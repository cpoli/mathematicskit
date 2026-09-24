r"""Convex hull: via :class:`scipy.spatial.ConvexHull` (Qhull), plus a
hand-rolled Graham scan kept only for pedagogical comparison in 2D.

``scipy.spatial.ConvexHull`` wraps Qhull, a robust, well-tested
computational-geometry library that handles degenerate/collinear input
correctly in any dimension; mathematicskit does not reimplement it as the
primary API. See de Berg et al., *Computational Geometry: Algorithms and
Applications*, 3rd ed., Ch. 1 (for Graham scan) and O'Rourke,
*Computational Geometry in C*, 2nd ed., Ch. 4 (for the general
:math:`d`-dimensional problem Qhull solves).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull as _ScipyConvexHull

from mathematicskit.geometry.core.base import ConvexHullResult

__all__ = ["convex_hull", "graham_scan"]


def convex_hull(points: np.ndarray) -> ConvexHullResult:
    r"""Convex hull of a point set, via :class:`scipy.spatial.ConvexHull` (Qhull).

    Works in any dimension; the 2D case returns hull vertices in
    counterclockwise order. See O'Rourke, *Computational Geometry in
    C*, 2nd ed., Ch. 4.

    Parameters
    ----------
    points : ndarray, shape (n, d)

    Returns
    -------
    ConvexHullResult

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    >>> result = convex_hull(points)
    >>> sorted(int(v) for v in result.vertices)  # the interior point (0.5, 0.5) is excluded
    [0, 1, 2, 3]
    >>> round(result.volume, 6)  # area of the unit square
    1.0
    """
    hull = _ScipyConvexHull(points)
    return ConvexHullResult(
        points=np.asarray(points), vertices=hull.vertices, simplices=hull.simplices, volume=float(hull.volume), area=float(hull.area), method="qhull"
    )


def graham_scan(points: np.ndarray) -> ConvexHullResult:
    r"""Convex hull of a *planar* point set via the Graham scan.

    Sorts points by angle around the lowest (then leftmost) point, and
    sweeps them, popping the hull-in-progress whenever the last three
    points make a clockwise (non-left) turn -- kept hand-rolled
    specifically to compare against :func:`convex_hull`'s Qhull-based
    result, not as the primary 2D convex-hull API. See de Berg et al.,
    *Computational Geometry*, 3rd ed., Ch. 1.1.

    Parameters
    ----------
    points : ndarray, shape (n, 2)

    Returns
    -------
    ConvexHullResult
        ``vertices`` holds the hull's point indices in counterclockwise
        order. ``simplices`` is left empty, since the Graham scan produces
        only the ordered vertex cycle, not a facet list. ``volume`` is the
        enclosed area via the shoelace formula on those vertices --
        matching :func:`convex_hull`, where Qhull likewise reports the
        enclosed area as ``volume`` in 2D. ``area`` is left at ``0.0``:
        Qhull uses it for the hull's perimeter, which this scan does not
        compute.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    >>> result = graham_scan(points)
    >>> sorted(int(v) for v in result.vertices)
    [0, 1, 2, 3]
    >>> round(result.volume, 6)
    1.0
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points")

    start_idx = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    start = points[start_idx]

    def polar_key(i):
        dx, dy = points[i, 0] - start[0], points[i, 1] - start[1]
        angle = np.arctan2(dy, dx)
        dist = dx * dx + dy * dy
        return (angle, dist)

    order = sorted((i for i in range(n) if i != start_idx), key=polar_key)
    order = [start_idx] + order

    def cross(o, a, b):
        return (points[a, 0] - points[o, 0]) * (points[b, 1] - points[o, 1]) - (points[a, 1] - points[o, 1]) * (points[b, 0] - points[o, 0])

    stack = []
    for i in order:
        while len(stack) >= 2 and cross(stack[-2], stack[-1], i) <= 0:
            stack.pop()
        stack.append(i)

    from mathematicskit.geometry.systems.polygon import polygon_area

    hull_points = points[stack]
    area = polygon_area(hull_points) if len(stack) >= 3 else 0.0
    return ConvexHullResult(points=points, vertices=np.array(stack), simplices=np.empty((0, 2), dtype=np.int64), volume=area, area=0.0, method="graham_scan")
