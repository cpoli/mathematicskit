r"""Minimum spanning tree: Kruskal's algorithm (via scipy), and a
hand-rolled Prim's algorithm kept for comparison.

``scipy.sparse.csgraph.minimum_spanning_tree`` (Kruskal-based, using a
disjoint-set/union-find internally) is the primary API here; a
from-scratch Prim's implementation is kept alongside it specifically to
*compare* the two algorithms' approach (Kruskal grows a forest by
globally cheapest edge, Prim grows a single tree by locally cheapest
edge) -- not as a competing primary API. See Cormen et al.,
*Introduction to Algorithms*, 3rd ed., Ch. 23.
"""

from __future__ import annotations

import heapq

from scipy.sparse import csgraph

from mathkit.graph_theory.core.base import Graph, MSTResult

__all__ = ["kruskal_mst", "prim_mst"]


def kruskal_mst(graph: Graph) -> MSTResult:
    r"""Minimum spanning tree via Kruskal's algorithm.

    Via :func:`scipy.sparse.csgraph.minimum_spanning_tree`, which sorts
    edges by weight and adds each one that doesn't close a cycle
    (tracked internally with a disjoint-set structure) until every
    vertex is connected. See Cormen et al., *Introduction to Algorithms*,
    3rd ed., Ch. 23.2.

    Parameters
    ----------
    graph : Graph
        Undirected (the notion of a spanning tree assumes this).

    Returns
    -------
    MSTResult

    Examples
    --------
    >>> g = Graph(4)
    >>> g.add_edge(0, 1, 1.0)
    >>> g.add_edge(1, 2, 2.0)
    >>> g.add_edge(2, 3, 3.0)
    >>> g.add_edge(0, 3, 10.0)
    >>> g.add_edge(0, 2, 4.0)
    >>> result = kruskal_mst(g)
    >>> result.total_weight
    6.0
    """
    mst_sparse = csgraph.minimum_spanning_tree(graph.to_sparse())
    coo = mst_sparse.tocoo()
    edges = [(int(u), int(v), float(w)) for u, v, w in zip(coo.row, coo.col, coo.data)]
    return MSTResult(edges=edges, total_weight=float(coo.data.sum()), method="kruskal")


def prim_mst(graph: Graph, start: int = 0) -> MSTResult:
    r"""Minimum spanning tree via Prim's algorithm (hand-rolled).

    Grows a single tree from `start`, at each step adding the cheapest
    edge leaving the current tree to a not-yet-included vertex (via a
    binary min-heap of candidate edges) -- unlike Kruskal, Prim always
    maintains one connected tree rather than a forest. Kept hand-rolled
    specifically to compare against :func:`kruskal_mst`, not as the
    primary MST API. See Cormen et al., *Introduction to Algorithms*,
    3rd ed., Ch. 23.2.

    Parameters
    ----------
    graph : Graph
        Undirected and connected.
    start : int
        Starting vertex.

    Returns
    -------
    MSTResult

    Examples
    --------
    >>> g = Graph(4)
    >>> g.add_edge(0, 1, 1.0)
    >>> g.add_edge(1, 2, 2.0)
    >>> g.add_edge(2, 3, 3.0)
    >>> g.add_edge(0, 3, 10.0)
    >>> g.add_edge(0, 2, 4.0)
    >>> result = prim_mst(g)
    >>> result.total_weight
    6.0
    """
    in_tree = [False] * graph.n_vertices
    in_tree[start] = True
    edges = []
    total_weight = 0.0
    heap = [(w, start, v) for v, w in graph.neighbors(start).items()]
    heapq.heapify(heap)

    while heap and len(edges) < graph.n_vertices - 1:
        w, u, v = heapq.heappop(heap)
        if in_tree[v]:
            continue
        in_tree[v] = True
        edges.append((u, v, w))
        total_weight += w
        for next_v, next_w in graph.neighbors(v).items():
            if not in_tree[next_v]:
                heapq.heappush(heap, (next_w, v, next_v))

    return MSTResult(edges=edges, total_weight=total_weight, method="prim")
