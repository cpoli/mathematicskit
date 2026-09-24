r"""Small graph-generator helpers -- supporting numerics for building
test/example graphs, not algorithms in their own right.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.graph_theory.core.base import Graph

__all__ = ["complete_graph", "cycle_graph", "path_graph", "random_graph", "dodecahedron_graph", "grid_graph", "complete_bipartite_graph"]


def complete_graph(n: int) -> Graph:
    """The complete graph :math:`K_n`: every pair of vertices connected.

    Parameters
    ----------
    n : int

    Returns
    -------
    Graph

    Examples
    --------
    >>> g = complete_graph(4)
    >>> len(g.edges())
    6
    """
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def cycle_graph(n: int) -> Graph:
    """The cycle graph :math:`C_n`: vertices ``0, ..., n-1`` in a ring.

    Parameters
    ----------
    n : int

    Returns
    -------
    Graph

    Examples
    --------
    >>> g = cycle_graph(5)
    >>> len(g.edges())
    5
    """
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def path_graph(n: int) -> Graph:
    """The path graph :math:`P_n`: vertices ``0, ..., n-1`` in a line.

    Parameters
    ----------
    n : int

    Returns
    -------
    Graph

    Examples
    --------
    >>> g = path_graph(5)
    >>> len(g.edges())
    4
    """
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def random_graph(n: int, p: float, seed: int = 0, directed: bool = False) -> Graph:
    r"""An Erdős-Rényi random graph :math:`G(n, p)`: each possible edge present independently with probability ``p``.

    Parameters
    ----------
    n : int
    p : float
        Edge probability, ``0 <= p <= 1``.
    seed : int
    directed : bool

    Returns
    -------
    Graph

    Examples
    --------
    >>> g = random_graph(10, p=0.3, seed=0)
    >>> 0 <= len(g.edges()) <= 45
    True
    """
    rng = np.random.default_rng(seed)
    g = Graph(n, directed=directed)
    pairs = [(i, j) for i in range(n) for j in (range(n) if directed else range(i + 1, n))]
    for i, j in pairs:
        if rng.random() < p:
            g.add_edge(i, j)
    return g


def dodecahedron_graph() -> Graph:
    r"""The 20-vertex, 30-edge graph of the regular dodecahedron, the board of Hamilton's icosian game.

    Vertices 0-4 form the outer pentagon, 5-9 and 10-14 the two middle
    rings, and 15-19 the inner pentagon.

    Returns
    -------
    Graph

    Examples
    --------
    >>> g = dodecahedron_graph()
    >>> g.n_vertices, len(g.edges())
    (20, 30)
    """
    g = Graph(20)
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)  # outer pentagon
        g.add_edge(i, 5 + 2 * i)  # spokes to the middle ring
        g.add_edge(15 + i, 15 + (i + 1) % 5)  # inner pentagon
        g.add_edge(15 + i, 6 + 2 * i)  # spokes to the middle ring
    for k in range(10):
        g.add_edge(5 + k, 5 + (k + 1) % 10)  # middle decagon
    return g


def grid_graph(rows: int, cols: int, blocked=()) -> tuple:
    r"""A 4-connected grid graph with unit edge weights, and the planar position of each vertex.

    Vertex ``r * cols + c`` sits at position ``(c, r)``. Cells listed in
    ``blocked`` (as ``(row, col)`` pairs) get no edges.

    Parameters
    ----------
    rows, cols : int
    blocked : iterable of (int, int)

    Returns
    -------
    tuple of (Graph, ndarray)
        The graph and an array of shape ``(rows * cols, 2)`` of positions.

    Examples
    --------
    >>> g, pos = grid_graph(2, 3)
    >>> len(g.edges()), pos[4].tolist()
    (7, [1.0, 1.0])
    """
    blocked = set(blocked)
    g = Graph(rows * cols)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < rows and cc < cols and (rr, cc) not in blocked:
                    g.add_edge(r * cols + c, rr * cols + cc)
    positions = np.array([(c, r) for r in range(rows) for c in range(cols)], dtype=float)
    return g, positions


def complete_bipartite_graph(m: int, n: int) -> Graph:
    r"""The complete bipartite graph :math:`K_{m,n}`: vertices ``0..m-1`` joined to every vertex ``m..m+n-1``.

    Parameters
    ----------
    m, n : int

    Returns
    -------
    Graph

    Examples
    --------
    >>> len(complete_bipartite_graph(3, 4).edges())
    12
    """
    g = Graph(m + n)
    for i in range(m):
        for j in range(n):
            g.add_edge(i, m + j)
    return g
