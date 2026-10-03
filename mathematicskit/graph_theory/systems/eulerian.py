r"""Eulerian circuits and trails by Hierholzer's algorithm.

Euler (1736) showed that a walk using every edge exactly once needs at
most two odd-degree vertices; Hierholzer (1873) proved the converse for
connected graphs by construction. Start anywhere and follow unused
edges until stuck, which can only happen back at the start (or at the
other odd vertex); then splice in a detour from any vertex on the walk
that still has unused edges. The stack-based form below does this in
:math:`O(|E|)` time. Hand-rolled: there is no scipy equivalent. See C.
Hierholzer, "Ueber die Möglichkeit, einen Linienzug ohne Wiederholung
und ohne Unterbrechung zu umfahren," Mathematische Annalen 6 (1873),
30-32, and Fleischner, *Eulerian Graphs and Related Topics*, Part 1,
Vol. 2 (1991), Ch. X.
"""

from __future__ import annotations

from typing import Optional

from mathematicskit.graph_theory.core.base import Graph

__all__ = ["eulerian_circuit", "eulerian_trail", "odd_degree_vertices"]


def odd_degree_vertices(graph: Graph) -> list:
    """Vertices of odd degree in an undirected graph.

    Parameters
    ----------
    graph : Graph
        Undirected.

    Returns
    -------
    list of int

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import path_graph
    >>> odd_degree_vertices(path_graph(4))
    [0, 3]
    """
    return [v for v in range(graph.n_vertices) if len(graph.neighbors(v)) % 2 == 1]


def _hierholzer(graph: Graph, start: int) -> list:
    remaining = {u: list(graph.neighbors(u)) for u in range(graph.n_vertices)}
    used: set = set()
    stack, walk = [start], []
    while stack:
        u = stack[-1]
        while remaining[u]:
            v = remaining[u].pop()
            key = (u, v) if graph.directed else frozenset((u, v))
            if key not in used:
                used.add(key)
                stack.append(v)
                break
        else:
            walk.append(stack.pop())
    return walk[::-1]


def _n_edges(graph: Graph) -> int:
    return len(graph.edges())


def eulerian_trail(graph: Graph, start: Optional[int] = None) -> Optional[list]:
    r"""A walk using every edge exactly once, or ``None`` if there is none.

    For an undirected graph whose edges lie in one connected component,
    a trail exists iff zero or two vertices have odd degree. With two, the
    trail runs from one odd vertex to the other; with none, it is a
    closed circuit. A directed graph needs every vertex to have equal
    in- and out-degree (a circuit), or exactly one vertex with one extra
    out-edge and one with one extra in-edge (an open trail).

    Parameters
    ----------
    graph : Graph
    start : int, optional
        Starting vertex. Must be an odd vertex (undirected) or the vertex
        with an extra out-edge (directed) when the trail is open; defaults
        to that vertex, or to the lowest-numbered vertex with an edge.

    Returns
    -------
    list of int or None
        The vertices in order, ``len(edges) + 1`` of them; for a circuit
        the first and last are equal.

    Examples
    --------
    >>> from mathematicskit.graph_theory.core.base import Graph
    >>> house = Graph(5)  # "das Haus vom Nikolaus": a square, its diagonals, and a roof
    >>> for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (2, 4), (3, 4)]:
    ...     house.add_edge(u, v)
    >>> trail = eulerian_trail(house)
    >>> len(trail), sorted([trail[0], trail[-1]])
    (9, [0, 1])
    """
    n = graph.n_vertices
    if graph.directed:
        out_deg = [len(graph.neighbors(u)) for u in range(n)]
        in_deg = [0] * n
        for _u, v, _w in graph.edges():
            in_deg[v] += 1
        surplus = [out_deg[v] - in_deg[v] for v in range(n)]
        if any(abs(s) > 1 for s in surplus) or sum(1 for s in surplus if s == 1) > 1:
            return None
        sources = [v for v in range(n) if surplus[v] == 1]
    else:
        odd = odd_degree_vertices(graph)
        if len(odd) not in (0, 2):
            return None
        sources = odd
    with_edges = [v for v in range(n) if graph.neighbors(v)]
    if not with_edges:
        return [0 if start is None else start] if n else None
    if start is None:
        start = sources[0] if sources else with_edges[0]
    elif sources and start not in sources:
        return None
    walk = _hierholzer(graph, start)
    return walk if len(walk) == _n_edges(graph) + 1 else None  # shorter: the edges are not all connected


def eulerian_circuit(graph: Graph, start: Optional[int] = None) -> Optional[list]:
    r"""A closed walk using every edge exactly once, or ``None`` if there is none.

    Exists iff every vertex has even degree (in-degree equal to
    out-degree, for a directed graph) and the edges lie in one connected
    component: Euler's necessary condition, which Hierholzer proved
    sufficient.

    Parameters
    ----------
    graph : Graph
    start : int, optional
        Starting (and ending) vertex; defaults to the lowest-numbered
        vertex with an edge.

    Returns
    -------
    list of int or None
        ``len(edges) + 1`` vertices, beginning and ending at ``start``.

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import complete_graph
    >>> circuit = eulerian_circuit(complete_graph(5))  # every vertex has degree 4
    >>> len(circuit), circuit[0] == circuit[-1]
    (11, True)
    >>> eulerian_circuit(complete_graph(4)) is None  # every vertex has degree 3
    True
    """
    trail = eulerian_trail(graph, start)
    return trail if trail is not None and trail[0] == trail[-1] else None
