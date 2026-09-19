r"""Maximum flow / minimum cut, via :mod:`scipy.sparse.csgraph`.

``scipy.sparse.csgraph.maximum_flow`` implements Dinic's algorithm (or
Edmonds-Karp) directly; mathkit does not reimplement augmenting-path
search. The corresponding minimum cut is derived from the max-flow
result's residual graph -- a standard, simple bookkeeping step (a single
reachability search), not a reimplementation of the flow algorithm
itself. See Cormen et al., *Introduction to Algorithms*, 3rd ed.,
Ch. 26, and the max-flow min-cut theorem (Ch. 26.2).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csgraph

from mathkit.graph_theory.core.base import Graph, MaxFlowResult

__all__ = ["max_flow_min_cut"]


def max_flow_min_cut(graph: Graph, source: int, sink: int, method: str = "dinic") -> MaxFlowResult:
    r"""Maximum flow from `source` to `sink`, and the corresponding minimum cut.

    Edge weights are used as integer capacities (required by
    :func:`scipy.sparse.csgraph.maximum_flow`, which needs integer-valued
    capacities). The minimum cut is found by a reachability search from
    `source` in the residual graph (``capacity - flow``): by the
    max-flow min-cut theorem, the set of edges from reachable to
    unreachable vertices is a minimum cut, with total capacity equal to
    the maximum flow value. See Cormen et al., *Introduction to
    Algorithms*, 3rd ed., Ch. 26.2, Theorem 26.6.

    Parameters
    ----------
    graph : Graph
        Directed, with positive integer edge weights (capacities).
    source, sink : int
    method : {"dinic", "edmonds_karp"}
        Forwarded to :func:`scipy.sparse.csgraph.maximum_flow`.

    Returns
    -------
    MaxFlowResult

    Examples
    --------
    >>> g = Graph(4, directed=True)
    >>> g.add_edge(0, 1, 3)
    >>> g.add_edge(0, 2, 2)
    >>> g.add_edge(1, 3, 2)
    >>> g.add_edge(2, 3, 3)
    >>> g.add_edge(1, 2, 1)
    >>> result = max_flow_min_cut(g, source=0, sink=3)
    >>> result.flow_value
    5.0
    """
    capacities = graph.to_sparse().astype(np.int64)
    scipy_result = csgraph.maximum_flow(capacities, source, sink, method=method)
    flow_matrix = np.asarray(scipy_result.flow.toarray(), dtype=np.float64)
    capacity_dense = np.asarray(capacities.toarray(), dtype=np.float64)

    residual = capacity_dense - flow_matrix
    n = graph.n_vertices
    visited = np.zeros(n, dtype=bool)
    visited[source] = True
    stack = [source]
    while stack:
        u = stack.pop()
        for v in range(n):
            if not visited[v] and residual[u, v] > 0:
                visited[v] = True
                stack.append(v)
    source_side = np.flatnonzero(visited)
    sink_side = np.flatnonzero(~visited)

    return MaxFlowResult(flow_value=float(scipy_result.flow_value), flow_matrix=flow_matrix, min_cut=(source_side, sink_side))
