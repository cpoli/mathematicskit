r"""Polyline simplification by the Ramer-Douglas-Peucker algorithm.

Hand-rolled (no numpy/scipy equivalent). See D. H. Douglas and T. K.
Peucker, "Algorithms for the Reduction of the Number of Points Required
to Represent a Digitized Line or Its Caricature," The Canadian
Cartographer 10(2) (1973), 112-122, and U. Ramer, "An Iterative
Procedure for the Polygonal Approximation of Plane Curves," Computer
Graphics and Image Processing 1(3) (1972), 244-256.
"""

from __future__ import annotations

import numpy as np

__all__ = ["douglas_peucker"]


def _distances_to_segment(points, a, b):
    ab = b - a
    length2 = ab @ ab
    if length2 == 0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip((points - a) @ ab / length2, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def douglas_peucker(points, epsilon: float) -> np.ndarray:
    r"""Simplify a polyline so that no removed point lies farther than ``epsilon`` from the result.

    Keeps the two endpoints, finds the interior point farthest from the
    segment joining them, and recurses on both halves if that distance
    exceeds ``epsilon``; otherwise drops every interior point.

    Parameters
    ----------
    points : array_like, shape (n, 2)
    epsilon : float
        Tolerance, in the same units as the points.

    Returns
    -------
    ndarray, shape (m, 2)
        The retained vertices, a subset of ``points`` in their original order.

    Examples
    --------
    >>> douglas_peucker([[0, 0], [1, 0.1], [2, -0.1], [3, 5], [4, 6], [5, 7]], 1.0).tolist()
    [[0.0, 0.0], [2.0, -0.1], [3.0, 5.0], [5.0, 7.0]]
    """
    pts = np.asarray(points, dtype=float)
    keep = np.zeros(len(pts), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        d = _distances_to_segment(pts[first + 1 : last], pts[first], pts[last])
        k = int(np.argmax(d))
        if d[k] > epsilon:
            split = first + 1 + k
            keep[split] = True
            stack.extend([(first, split), (split, last)])
    return pts[keep]
