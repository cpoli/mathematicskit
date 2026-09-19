r"""Small graph-generator helpers -- supporting numerics for building
test/example graphs, not algorithms in their own right.
"""

from __future__ import annotations

import numpy as np

from mathkit.graph_theory.core.base import Graph

__all__ = ["complete_graph", "cycle_graph", "path_graph", "random_graph"]


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
    r"""An Erdos-Renyi random graph :math:`G(n, p)`: each possible edge present independently with probability ``p``.

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
