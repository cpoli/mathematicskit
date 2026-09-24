r"""Hamiltonian cycles by backtracking search.

Deciding whether a Hamiltonian cycle exists is NP-complete, so there is
no efficient library routine; this exhaustive search with simple pruning
is feasible only for small graphs. See W. R. Hamilton, "Memorandum
Respecting a New System of Roots of Unity," Philosophical Magazine
Series 4, 12 (1856), 446, and M. R. Garey and D. S. Johnson, *Computers
and Intractability* (San Francisco: W. H. Freeman, 1979), Sec. 3.1.4.
"""

from __future__ import annotations

from typing import Optional

from mathematicskit.graph_theory.core.base import Graph

__all__ = ["hamiltonian_cycle"]


def hamiltonian_cycle(graph: Graph, start: int = 0) -> Optional[list]:
    r"""A cycle through every vertex exactly once, or ``None`` if there is none.

    Extends a path one vertex at a time, trying unvisited neighbours in
    order of fewest remaining unvisited neighbours (Warnsdorff's
    heuristic), and backtracks at dead ends.

    Parameters
    ----------
    graph : Graph
        Undirected.
    start : int

    Returns
    -------
    list of int or None
        The cycle's vertices, beginning at ``start``; the closing edge back
        to ``start`` is implied.

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import dodecahedron_graph, complete_bipartite_graph
    >>> len(hamiltonian_cycle(dodecahedron_graph()))  # Hamilton's icosian game
    20
    >>> hamiltonian_cycle(complete_bipartite_graph(2, 3)) is None  # unbalanced bipartite
    True
    """
    n = graph.n_vertices
    path = [start]
    visited = [False] * n
    visited[start] = True

    def unvisited_degree(v):
        return sum(1 for w in graph.neighbors(v) if not visited[w])

    def extend() -> bool:
        if len(path) == n:
            return start in graph.neighbors(path[-1])
        for v in sorted((w for w in graph.neighbors(path[-1]) if not visited[w]), key=unvisited_degree):
            visited[v] = True
            path.append(v)
            if extend():
                return True
            path.pop()
            visited[v] = False
        return False

    return list(path) if extend() else None
