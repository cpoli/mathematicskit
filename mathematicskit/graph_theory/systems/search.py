r"""A* search: shortest paths guided by a heuristic.

Hand-rolled with :mod:`heapq`: SciPy's graph routines have no
heuristic-guided search. See P. E. Hart, N. J. Nilsson, and B. Raphael,
"A Formal Basis for the Heuristic Determination of Minimum Cost Paths,"
IEEE Transactions on Systems Science and Cybernetics 4(2) (1968),
100-107.
"""

from __future__ import annotations

import heapq
from typing import Callable, Optional

from mathematicskit.graph_theory.core.base import Graph, SearchResult

__all__ = ["astar_shortest_path"]


def astar_shortest_path(graph: Graph, source: int, target: int, heuristic: Optional[Callable[[int], float]] = None) -> SearchResult:
    r"""Shortest path from ``source`` to ``target``, expanding vertices in order of :math:`g(v) + h(v)`.

    :math:`g` is the best known distance from the source and :math:`h`
    an estimate of the remaining distance. If :math:`h` never
    overestimates (it is *admissible*) and is consistent, the first time
    the target is expanded its distance is optimal. With :math:`h = 0`
    the search is Dijkstra's algorithm.

    Parameters
    ----------
    graph : Graph
        Non-negative edge weights.
    source, target : int
    heuristic : callable, optional
        ``heuristic(v)`` estimates the distance from ``v`` to ``target``;
        defaults to 0.

    Returns
    -------
    SearchResult

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import grid_graph
    >>> g, pos = grid_graph(5, 5)
    >>> manhattan = lambda v: abs(pos[v] - pos[24]).sum()
    >>> result = astar_shortest_path(g, 0, 24, manhattan)
    >>> result.distance, len(result.path)
    (8.0, 9)
    """
    h = heuristic or (lambda v: 0.0)
    best = {source: 0.0}
    parent = {source: None}
    queue = [(h(source), 0.0, source)]
    closed = set()
    expanded = 0
    while queue:
        _, g_u, u = heapq.heappop(queue)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == target:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return SearchResult(path=path[::-1], distance=g_u, n_expanded=expanded)
        for v, w in graph.neighbors(u).items():
            candidate = g_u + w
            if candidate < best.get(v, float("inf")):
                best[v] = candidate
                parent[v] = u
                heapq.heappush(queue, (candidate + h(v), candidate, v))
    return SearchResult(path=[], distance=float("inf"), n_expanded=expanded)
