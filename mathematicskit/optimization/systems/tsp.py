r"""The travelling salesman problem: nearest-neighbour and 2-opt heuristics, and
the exact Held-Karp dynamic program.

Find the shortest closed tour visiting each of :math:`n` cities once.
There are :math:`(n-1)!/2` tours, and the problem is NP-hard. Held and
Karp (1962) cut exhaustive search to :math:`O(n^2 2^n)` with a dynamic
program over subsets, still the best exact worst-case bound known; it is
practical to about 20 cities. Heuristics trade optimality for speed:
nearest neighbour builds a tour greedily, and Croes's (1958) 2-opt move
repairs it by reversing a segment whenever that removes a crossing.
Hand-rolled: scipy has no TSP solver. See M. Held and R. M. Karp, "A
Dynamic Programming Approach to Sequencing Problems," Journal of the
Society for Industrial and Applied Mathematics 10(1) (1962), 196-210;
G. A. Croes, "A Method for Solving Traveling-Salesman Problems,"
Operations Research 6(6) (1958), 791-812; and Applegate, Bixby, Chvátal
and Cook, *The Traveling Salesman Problem: A Computational Study*
(2006).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mathematicskit.optimization.core.base import TourResult

__all__ = ["distance_matrix", "tour_length", "tsp_nearest_neighbor", "tsp_two_opt", "tsp_held_karp"]


def distance_matrix(points: np.ndarray) -> np.ndarray:
    """Euclidean distances between every pair of points.

    Parameters
    ----------
    points : array-like, shape (n, d)

    Returns
    -------
    ndarray, shape (n, n)

    Examples
    --------
    >>> distance_matrix([[0.0, 0.0], [3.0, 4.0]])
    array([[0., 5.],
           [5., 0.]])
    """
    p = np.asarray(points, dtype=np.float64)
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=-1))


def tour_length(dist: np.ndarray, tour) -> float:
    """Length of the closed tour ``tour[0] -> ... -> tour[-1] -> tour[0]``.

    Parameters
    ----------
    dist : ndarray, shape (n, n)
    tour : sequence of int

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> square = distance_matrix([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> tour_length(square, [0, 1, 2, 3]), round(tour_length(square, [0, 2, 1, 3]), 4)
    (4.0, 4.8284)
    """
    t = np.asarray(tour)
    return float(np.asarray(dist)[t, np.roll(t, -1)].sum())


def tsp_nearest_neighbor(dist: np.ndarray, start: int = 0) -> TourResult:
    r"""Greedy tour: from each city, go to the nearest city not yet visited.

    :math:`O(n^2)`. On Euclidean instances its tours are typically about
    25% longer than optimal, but it can be a factor
    :math:`\Theta(\log n)` worse (Rosenkrantz, Stearns and Lewis, 1977).

    Parameters
    ----------
    dist : ndarray, shape (n, n)
    start : int

    Returns
    -------
    TourResult

    Examples
    --------
    >>> pts = [[0, 0], [1, 0], [2, 0], [2, 1], [0, 1]]
    >>> tsp_nearest_neighbor(distance_matrix(pts)).tour
    [0, 1, 2, 3, 4]
    """
    dist = np.asarray(dist, dtype=np.float64)
    n = dist.shape[0]
    unvisited = np.ones(n, dtype=bool)
    tour = [int(start)]
    unvisited[start] = False
    for _ in range(n - 1):
        d = np.where(unvisited, dist[tour[-1]], np.inf)
        nxt = int(np.argmin(d))
        tour.append(nxt)
        unvisited[nxt] = False
    return TourResult(tour=tour, length=tour_length(dist, tour), method="nearest_neighbor")


def tsp_two_opt(dist: np.ndarray, tour: Optional[list] = None, max_passes: int = 1000) -> TourResult:
    r"""Improve a tour with 2-opt moves until none shortens it.

    A 2-opt move deletes edges :math:`(t_i, t_{i+1})` and
    :math:`(t_j, t_{j+1})` and reconnects the tour by reversing the
    segment :math:`t_{i+1} \dots t_j`. It shortens the tour iff
    :math:`d(t_i, t_j) + d(t_{i+1}, t_{j+1}) < d(t_i, t_{i+1}) + d(t_j, t_{j+1})`;
    in the plane every crossing can be removed this way, so a 2-optimal
    Euclidean tour never crosses itself.

    Parameters
    ----------
    dist : ndarray, shape (n, n)
    tour : list of int, optional
        Starting tour; defaults to :func:`tsp_nearest_neighbor`'s.
    max_passes : int
        Maximum number of sweeps over all pairs ``(i, j)``.

    Returns
    -------
    TourResult
        ``history`` records the length after each improving move.

    Examples
    --------
    >>> square = distance_matrix([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> result = tsp_two_opt(square, [0, 2, 1, 3])  # a crossed "bow tie"
    >>> result.length
    4.0
    """
    dist = np.asarray(dist, dtype=np.float64)
    t = list(tsp_nearest_neighbor(dist).tour if tour is None else tour)
    n = len(t)
    length = tour_length(dist, t)
    history = [length]
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                gain = dist[a, b] + dist[c, d] - dist[a, c] - dist[b, d]
                if gain > 1e-12:
                    t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
                    length -= gain
                    history.append(length)
                    improved = True
        if not improved:
            break
    return TourResult(tour=t, length=tour_length(dist, t), method="two_opt", history=history)


def tsp_held_karp(dist: np.ndarray) -> TourResult:
    r"""The optimal tour, by the Held-Karp dynamic program.

    Let :math:`C(S, j)` be the length of the shortest path that starts at
    city 0, visits every city of :math:`S \subseteq \{1, \dots, n-1\}`
    once, and ends at :math:`j \in S`. Then

    .. math::

       C(\{j\}, j) = d_{0j}, \qquad
       C(S, j) = \min_{i \in S \setminus \{j\}} C(S \setminus \{j\}, i) + d_{ij},

    and the optimal tour length is
    :math:`\min_j C(\{1, \dots, n-1\}, j) + d_{j0}`. Subsets are bitmasks,
    and each table row is updated with one vectorized minimum, for
    :math:`O(n^2 2^n)` time and :math:`O(n 2^n)` memory.

    Parameters
    ----------
    dist : ndarray, shape (n, n)
        Need not be symmetric. Keep ``n`` below about 20.

    Returns
    -------
    TourResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> d = distance_matrix(rng.uniform(size=(8, 2)))
    >>> from itertools import permutations
    >>> brute = min(tour_length(d, (0,) + p) for p in permutations(range(1, 8)))
    >>> bool(np.isclose(tsp_held_karp(d).length, brute))
    True
    """
    dist = np.asarray(dist, dtype=np.float64)
    n = dist.shape[0]
    if n <= 2:
        tour = list(range(n))
        return TourResult(tour=tour, length=tour_length(dist, tour) if n else 0.0, method="held_karp")
    m = n - 1  # cities 1..n-1 are bits 0..m-1
    full = 1 << m
    cost = np.full((full, m), np.inf)
    parent = np.full((full, m), -1, dtype=np.int64)
    for j in range(m):
        cost[1 << j, j] = dist[0, j + 1]
    sub = dist[1:, 1:]
    for mask in range(1, full):
        row = cost[mask]
        if not np.isfinite(row).any():
            continue
        candidates = row[:, None] + sub  # candidates[i, j]: end at i, then go to j
        for j in range(m):
            bit = 1 << j
            if mask & bit:
                continue
            i = int(np.argmin(candidates[:, j]))
            value = candidates[i, j]
            new = mask | bit
            if value < cost[new, j]:
                cost[new, j] = value
                parent[new, j] = i
    closing = cost[full - 1] + dist[1:, 0]
    j = int(np.argmin(closing))
    tour, mask = [], full - 1
    while j != -1:
        tour.append(j + 1)
        j, mask = int(parent[mask, j]), mask & ~(1 << j)
    tour = [0] + tour[::-1]
    return TourResult(tour=tour, length=float(closing.min()), method="held_karp")
