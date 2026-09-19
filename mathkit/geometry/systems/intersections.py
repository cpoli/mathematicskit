r"""Line/segment intersection and point-in-polygon tests.

No direct scipy equivalent for either -- these are elementary
computational-geometry primitives with no dedicated library routine.
See de Berg et al., *Computational Geometry*, 3rd ed., Ch. 2
(segment intersection) and O'Rourke, *Computational Geometry in C*,
2nd ed., Ch. 7 (point-in-polygon).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["segment_intersection", "point_in_polygon"]


def segment_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    r"""Intersection point of segments ``p1-p2`` and ``p3-p4``, if they cross.

    Writes both segments parametrically (:math:`p_1 + t(p_2-p_1)`,
    :math:`p_3 + u(p_4-p_3)`) and solves the resulting :math:`2\times2`
    linear system for :math:`t, u \in [0, 1]` via Cramer's rule; returns
    ``None`` if the segments are parallel or don't overlap within their
    endpoints. See de Berg et al., *Computational Geometry*, 3rd ed.,
    Ch. 2.

    Parameters
    ----------
    p1, p2 : ndarray, shape (2,)
        Endpoints of the first segment.
    p3, p4 : ndarray, shape (2,)
        Endpoints of the second segment.

    Returns
    -------
    ndarray, shape (2,), or None

    Examples
    --------
    >>> import numpy as np
    >>> p = segment_intersection(np.array([0.0, 0.0]), np.array([2.0, 2.0]), np.array([0.0, 2.0]), np.array([2.0, 0.0]))
    >>> np.allclose(p, [1.0, 1.0])
    True
    >>> segment_intersection(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])) is None
    True
    """
    p1, p2, p3, p4 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4))
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None  # parallel (or collinear)
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p1 + t * d1
    return None


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    r"""Whether `point` lies inside `polygon`, via the ray-casting (crossing-number) algorithm.

    Casts a ray from `point` in the :math:`+x` direction and counts how
    many polygon edges it crosses; an odd count means the point is
    inside (the standard Jordan-curve-theorem argument). See O'Rourke,
    *Computational Geometry in C*, 2nd ed., Ch. 7.4.

    Parameters
    ----------
    point : ndarray, shape (2,)
    polygon : ndarray, shape (n, 2)
        Vertices in order (either winding direction); the polygon
        implicitly closes back to its first vertex.

    Returns
    -------
    bool

    Examples
    --------
    >>> import numpy as np
    >>> square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    >>> point_in_polygon(np.array([2.0, 2.0]), square)
    True
    >>> point_in_polygon(np.array([5.0, 2.0]), square)
    False
    """
    x, y = point
    poly = np.asarray(polygon, dtype=np.float64)
    n = poly.shape[0]
    inside = False
    x1, y1 = poly[-1]
    for i in range(n):
        x2, y2 = poly[i]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside
