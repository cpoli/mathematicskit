r"""Graph coloring: a greedy heuristic and exact backtracking search.

No scipy equivalent -- graph coloring (deciding if ``k`` colors
suffice) is NP-complete in general, so scipy has no built-in solver.
See Cormen et al., *Introduction to Algorithms*, 3rd ed., Ch. 34
(NP-completeness), and West, *Introduction to Graph Theory*, 2nd ed.,
Ch. 5.
"""

from __future__ import annotations

from typing import Optional

from mathkit.graph_theory.core.base import ColoringResult

__all__ = ["greedy_coloring", "backtracking_coloring"]


def greedy_coloring(graph, order=None) -> ColoringResult:
    r"""Greedy graph coloring: color each vertex with the lowest color not used by its already-colored neighbors.

    Fast (:math:`O(V+E)`) but not optimal -- the number of colors used
    depends on vertex order and can exceed the graph's true chromatic
    number by an arbitrarily large factor in the worst case (though it's
    exact for common structured graphs like trees and bipartite graphs
    with a good order). See West, *Introduction to Graph Theory*, 2nd
    ed., Sec. 5.1.

    Parameters
    ----------
    graph : Graph
    order : sequence of int, optional
        Vertex processing order; defaults to ``0, 1, ..., n-1``.

    Returns
    -------
    ColoringResult

    Examples
    --------
    >>> from mathkit.graph_theory.core.base import Graph
    >>> g = Graph(4)
    >>> g.add_edge(0, 1)
    >>> g.add_edge(1, 2)
    >>> g.add_edge(2, 3)
    >>> g.add_edge(3, 0)
    >>> result = greedy_coloring(g)  # a 4-cycle is bipartite: 2 colors suffice
    >>> result.num_colors
    2
    """
    order = list(range(graph.n_vertices)) if order is None else list(order)
    coloring = {}
    for v in order:
        used = {coloring[u] for u in graph.neighbors(v) if u in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[v] = color
    return ColoringResult(coloring=coloring, num_colors=max(coloring.values()) + 1, method="greedy")


def backtracking_coloring(graph, max_colors: Optional[int] = None) -> ColoringResult:
    r"""Exact minimum graph coloring via backtracking search.

    Tries increasing values of ``k`` (number of colors) and searches
    exhaustively (with pruning: a vertex is only assigned a color not
    already used by an adjacent, already-colored vertex) for a valid
    ``k``-coloring, stopping at the first ``k`` that succeeds -- so the
    result is the graph's true chromatic number, at exponential
    worst-case cost. Feasible only for small graphs. See Cormen et al.,
    *Introduction to Algorithms*, 3rd ed., Ch. 34.5.1 (as an example
    NP-complete decision problem, 3-colorability).

    Parameters
    ----------
    graph : Graph
    max_colors : int, optional
        Upper bound on colors tried; defaults to `graph.n_vertices`
        (always sufficient, one color per vertex).

    Returns
    -------
    ColoringResult

    Examples
    --------
    >>> from mathkit.graph_theory.core.base import Graph
    >>> # A complete graph on 4 vertices (K4) needs exactly 4 colors.
    >>> g = Graph(4)
    >>> for i in range(4):
    ...     for j in range(i + 1, 4):
    ...         g.add_edge(i, j)
    >>> result = backtracking_coloring(g)
    >>> result.num_colors
    4
    """
    n = graph.n_vertices
    max_colors = n if max_colors is None else max_colors
    vertices = list(range(n))

    def _try_color(k: int) -> Optional[dict]:
        coloring: dict = {}

        def _backtrack(idx: int) -> bool:
            if idx == len(vertices):
                return True
            v = vertices[idx]
            used = {coloring[u] for u in graph.neighbors(v) if u in coloring}
            for color in range(k):
                if color not in used:
                    coloring[v] = color
                    if _backtrack(idx + 1):
                        return True
                    del coloring[v]
            return False

        return coloring if _backtrack(0) else None

    for k in range(1, max_colors + 1):
        result = _try_color(k)
        if result is not None:
            return ColoringResult(coloring=result, num_colors=k, method="backtracking")
    raise RuntimeError(f"no valid coloring found with up to {max_colors} colors")
