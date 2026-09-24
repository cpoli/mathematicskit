r"""The closest pair of points, by Shamos and Hoey's divide-and-conquer
algorithm.

Hand-rolled because the :math:`O(n \log n)` divide-and-conquer
recursion is the algorithm being illustrated; for production use a
k-d tree (:class:`scipy.spatial.cKDTree`) answers the same question and
is used as a cross-check in this module's tests. See M. I. Shamos and D.
Hoey, "Closest-Point Problems," in *16th Annual Symposium on
Foundations of Computer Science* (IEEE, 1975), 151-162, and Cormen et
al., *Introduction to Algorithms*, 3rd ed., Sec. 33.4.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.geometry.core.base import ClosestPairResult

__all__ = ["closest_pair"]


def closest_pair(points) -> ClosestPairResult:
    r"""The two closest points in a planar point set, in :math:`O(n \log n)` time.

    Sorts by :math:`x`, splits at the median, solves both halves, and
    then checks only points within the current best distance :math:`\delta`
    of the dividing line. Sorted by :math:`y`, each such point needs to be
    compared with at most seven of its successors.

    Parameters
    ----------
    points : array_like, shape (n, 2)
        At least two points.

    Returns
    -------
    ClosestPairResult

    Examples
    --------
    >>> result = closest_pair([[0, 0], [5, 5], [1, 1], [9, 0], [5.5, 5]])
    >>> result.indices, result.distance
    ((1, 4), 0.5)
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("need at least two points")
    by_x = sorted(range(len(pts)), key=lambda i: (pts[i, 0], pts[i, 1]))

    def dist(i, j):
        return float(np.hypot(*(pts[i] - pts[j])))

    def solve(idx):
        """Return (distance, pair, idx sorted by y) for the index list idx (sorted by x)."""
        if len(idx) <= 3:
            best = min(((dist(i, j), (i, j)) for a, i in enumerate(idx) for j in idx[a + 1 :]), default=(np.inf, None))
            return best[0], best[1], sorted(idx, key=lambda i: pts[i, 1])
        mid = len(idx) // 2
        x_mid = pts[idx[mid], 0]
        d_left, pair_left, left_y = solve(idx[:mid])
        d_right, pair_right, right_y = solve(idx[mid:])
        d, pair = (d_left, pair_left) if d_left <= d_right else (d_right, pair_right)
        merged, a, b = [], 0, 0
        while a < len(left_y) or b < len(right_y):
            if b == len(right_y) or (a < len(left_y) and pts[left_y[a], 1] <= pts[right_y[b], 1]):
                merged.append(left_y[a])
                a += 1
            else:
                merged.append(right_y[b])
                b += 1
        strip = [i for i in merged if abs(pts[i, 0] - x_mid) < d]
        for s, i in enumerate(strip):
            for j in strip[s + 1 : s + 8]:
                if pts[j, 1] - pts[i, 1] >= d:
                    break
                dij = dist(i, j)
                if dij < d:
                    d, pair = dij, (i, j)
        return d, pair, merged

    d, pair, _ = solve(by_x)
    return ClosestPairResult(indices=tuple(sorted(pair)), distance=d)
