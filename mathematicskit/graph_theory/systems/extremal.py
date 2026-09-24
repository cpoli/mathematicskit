r"""Turán's theorem: the densest graphs without a large clique.

Hand-rolled (no library equivalent). See P. Turán, "Egy gráfelméleti
szélsőértékfeladatról," Matematikai és Fizikai Lapok 48 (1941),
436-452, and M. Aigner and G. M. Ziegler, *Proofs from THE BOOK*, 6th
ed., Ch. 41.
"""

from __future__ import annotations

from mathematicskit.graph_theory.core.base import Graph

__all__ = ["turan_graph", "turan_number", "clique_number"]


def turan_graph(n: int, r: int) -> Graph:
    r"""The Turán graph :math:`T(n, r)`: ``n`` vertices split into ``r`` near-equal parts, joined across parts.

    Parameters
    ----------
    n : int
    r : int
        Number of parts, ``r >= 1``.

    Returns
    -------
    Graph

    Examples
    --------
    >>> len(turan_graph(6, 2).edges())  # K_{3,3}
    9
    """
    part = [i % r for i in range(n)]
    g = Graph(n)
    for u in range(n):
        for v in range(u + 1, n):
            if part[u] != part[v]:
                g.add_edge(u, v)
    return g


def turan_number(n: int, r: int) -> int:
    r"""The number of edges of :math:`T(n, r)`, the maximum for an ``n``-vertex graph with no :math:`K_{r+1}`.

    Parameters
    ----------
    n : int
    r : int

    Returns
    -------
    int

    Examples
    --------
    >>> turan_number(10, 2)  # Mantel (1907): floor(n^2 / 4) edges without a triangle
    25
    """
    q, s = divmod(n, r)
    sizes = [q + 1] * s + [q] * (r - s)
    return (n * n - sum(k * k for k in sizes)) // 2


def clique_number(graph: Graph) -> int:
    r"""The size of the largest clique, by Bron-Kerbosch search with pivoting.

    Exponential in the worst case; intended for small graphs.

    Parameters
    ----------
    graph : Graph
        Undirected.

    Returns
    -------
    int

    Examples
    --------
    >>> clique_number(turan_graph(9, 3))
    3
    """
    neighbours = {u: set(graph.neighbors(u)) - {u} for u in range(graph.n_vertices)}
    best = 0

    def expand(size, candidates, excluded):
        nonlocal best
        if not candidates and not excluded:
            best = max(best, size)
            return
        if size + len(candidates) <= best:
            return
        pivot = max(candidates | excluded, key=lambda u: len(neighbours[u] & candidates))
        for v in list(candidates - neighbours[pivot]):
            expand(size + 1, candidates & neighbours[v], excluded & neighbours[v])
            candidates = candidates - {v}
            excluded = excluded | {v}

    expand(0, set(range(graph.n_vertices)), set())
    return best
