r"""The smallest circle enclosing a set of points, by Welzl's randomized
algorithm.

Hand-rolled: SciPy has no minimum enclosing circle. James Joseph
Sylvester posed the problem in 1857; Emo Welzl's 1991 algorithm solves
it in expected linear time. See E. Welzl, "Smallest Enclosing Disks
(Balls and Ellipsoids)," in *New Results and New Trends in Computer
Science*, Lecture Notes in Computer Science 555 (Berlin: Springer,
1991), 359-370.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.geometry.core.base import CircleResult

__all__ = ["min_enclosing_circle"]


def _circle_two(a, b):
    center = (a + b) / 2
    return center, float(np.linalg.norm(a - center))


def _circle_three(a, b, c):
    bx, by = b - a
    cx, cy = c - a
    d = 2 * (bx * cy - by * cx)
    if abs(d) < 1e-14:
        pairs = [(a, b), (a, c), (b, c)]
        return max((_circle_two(p, q) for p, q in pairs), key=lambda circle: circle[1])
    ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d
    uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d
    center = a + np.array([ux, uy])
    return center, float(np.hypot(ux, uy))


def min_enclosing_circle(points, seed: int = 0) -> CircleResult:
    r"""The smallest circle containing every point, by Welzl's move-to-front algorithm.

    Processes the points in random order. Whenever a point falls outside
    the current circle, the optimal circle must pass through it, and the
    search restarts on the points seen so far with that point fixed on
    the boundary. The optimum is determined by at most three points.

    Parameters
    ----------
    points : array_like, shape (n, 2)
    seed : int
        Seed for the random processing order (the result does not depend on it).

    Returns
    -------
    CircleResult

    Examples
    --------
    >>> result = min_enclosing_circle([[0, 0], [2, 0], [1, 0.5]])
    >>> result.center.tolist(), result.radius
    ([1.0, 0.0], 1.0)
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.random.default_rng(seed).permutation(len(pts))]
    eps = 1e-12

    def inside(center, radius, p):
        return np.linalg.norm(p - center) <= radius * (1 + eps) + eps

    center, radius = pts[0], 0.0
    for i in range(1, len(pts)):
        if inside(center, radius, pts[i]):
            continue
        center, radius = pts[i], 0.0
        for j in range(i):
            if inside(center, radius, pts[j]):
                continue
            center, radius = _circle_two(pts[i], pts[j])
            for k in range(j):
                if not inside(center, radius, pts[k]):
                    center, radius = _circle_three(pts[i], pts[j], pts[k])
    return CircleResult(center=np.asarray(center, dtype=float), radius=float(radius))
