"""Clique enumeration for flag complexes (Vietoris-Rips).

A Vietoris-Rips complex is the clique complex of its 1-skeleton: a set
of points spans a simplex exactly when every pair does. The cliques are
grown one vertex at a time from each simplex's lower neighbours, as in
A. Zomorodian, "Fast construction of the Vietoris-Rips complex,"
Computers & Graphics 34(3) (2010), 263-271. Hand-rolled: scipy has no
clique enumeration.
"""

from __future__ import annotations

import numpy as np

__all__ = ["clique_simplices"]


def clique_simplices(adjacency, max_dim: int) -> list[tuple[int, ...]]:
    """All cliques of a graph with at most ``max_dim + 1`` vertices, as sorted vertex tuples.

    Parameters
    ----------
    adjacency : array_like of bool, shape (n, n)
        Symmetric adjacency matrix (the diagonal is ignored).
    max_dim : int
        Largest simplex dimension to return.

    Returns
    -------
    list of tuple of int

    Examples
    --------
    >>> square_with_diagonal = np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]], dtype=bool)
    >>> [s for s in clique_simplices(square_with_diagonal, 2) if len(s) == 3]
    [(0, 1, 2), (0, 2, 3)]
    """
    adj = np.asarray(adjacency, dtype=bool)
    n = len(adj)
    out: list[tuple[int, ...]] = []

    def expand(simplex: tuple[int, ...], candidates: np.ndarray):
        out.append(simplex)
        if len(simplex) > max_dim:
            return
        for i, u in enumerate(candidates):
            rest = candidates[i + 1 :]
            expand(simplex + (int(u),), rest[adj[u, rest]])

    for v in range(n):
        later = np.arange(v + 1, n)
        expand((v,), later[adj[v, later]])
    return out
