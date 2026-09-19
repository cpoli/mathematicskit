r"""Shortest-path algorithms, via :mod:`scipy.sparse.csgraph`.

``scipy.sparse.csgraph`` already implements Dijkstra, Bellman-Ford, and
Floyd-Warshall correctly and efficiently (in compiled Cython); mathkit
does not reimplement them. See Cormen et al., *Introduction to
Algorithms*, 3rd ed., Ch. 24 (single-source) and Ch. 25 (all-pairs).
"""

from __future__ import annotations

from scipy.sparse import csgraph

from mathkit.graph_theory.core.base import Graph, ShortestPathResult

__all__ = ["dijkstra_shortest_paths", "bellman_ford_shortest_paths", "floyd_warshall_shortest_paths"]


def dijkstra_shortest_paths(graph: Graph, sources=None) -> ShortestPathResult:
    r"""Single- or multi-source shortest paths via Dijkstra's algorithm.

    Requires non-negative edge weights. Via
    :func:`scipy.sparse.csgraph.dijkstra`. See Cormen et al.,
    *Introduction to Algorithms*, 3rd ed., Ch. 24.3.

    Parameters
    ----------
    graph : Graph
    sources : int or sequence of int, optional
        Defaults to every vertex (all-pairs).

    Returns
    -------
    ShortestPathResult

    Examples
    --------
    >>> g = Graph(4)
    >>> g.add_edge(0, 1, 1.0)
    >>> g.add_edge(1, 2, 2.0)
    >>> g.add_edge(0, 2, 5.0)
    >>> g.add_edge(2, 3, 1.0)
    >>> result = dijkstra_shortest_paths(g, sources=0)
    >>> result.distances
    array([0., 1., 3., 4.])
    """
    distances, predecessors = csgraph.dijkstra(graph.to_sparse(), directed=graph.directed, indices=sources, return_predecessors=True)
    return ShortestPathResult(distances=distances, predecessors=predecessors, method="dijkstra")


def bellman_ford_shortest_paths(graph: Graph, sources=None) -> ShortestPathResult:
    r"""Single- or multi-source shortest paths via Bellman-Ford.

    Unlike :func:`dijkstra_shortest_paths`, tolerates negative edge
    weights (raising if a negative-weight cycle is reachable, since
    shortest paths are then undefined). Via
    :func:`scipy.sparse.csgraph.bellman_ford`. See Cormen et al.,
    *Introduction to Algorithms*, 3rd ed., Ch. 24.1.

    Parameters
    ----------
    graph : Graph
    sources : int or sequence of int, optional

    Returns
    -------
    ShortestPathResult

    Examples
    --------
    >>> g = Graph(3, directed=True)
    >>> g.add_edge(0, 1, 4.0)
    >>> g.add_edge(0, 2, 5.0)
    >>> g.add_edge(1, 2, -2.0)  # a negative edge weight
    >>> result = bellman_ford_shortest_paths(g, sources=0)
    >>> result.distances
    array([0., 4., 2.])
    """
    distances, predecessors = csgraph.bellman_ford(graph.to_sparse(), directed=graph.directed, indices=sources, return_predecessors=True)
    return ShortestPathResult(distances=distances, predecessors=predecessors, method="bellman_ford")


def floyd_warshall_shortest_paths(graph: Graph) -> ShortestPathResult:
    r"""All-pairs shortest paths via Floyd-Warshall.

    :math:`O(n^3)` dynamic program over intermediate vertices --
    typically preferred over running Dijkstra/Bellman-Ford from every
    source when the graph is dense. Via
    :func:`scipy.sparse.csgraph.floyd_warshall`. See Cormen et al.,
    *Introduction to Algorithms*, 3rd ed., Ch. 25.2.

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    ShortestPathResult
        ``distances`` shape (n, n): all-pairs distance matrix.

    Examples
    --------
    >>> g = Graph(3)
    >>> g.add_edge(0, 1, 1.0)
    >>> g.add_edge(1, 2, 2.0)
    >>> result = floyd_warshall_shortest_paths(g)
    >>> result.distances
    array([[0., 1., 3.],
           [1., 0., 2.],
           [3., 2., 0.]])
    """
    distances, predecessors = csgraph.floyd_warshall(graph.to_sparse(), directed=graph.directed, return_predecessors=True)
    return ShortestPathResult(distances=distances, predecessors=predecessors, method="floyd_warshall")
