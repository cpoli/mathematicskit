r"""Maximum matchings and minimum vertex covers in bipartite graphs.

The matching itself comes from
:func:`scipy.sparse.csgraph.maximum_bipartite_matching`, which
implements the Hopcroft-Karp algorithm. mathematicskit adds the König
construction of a minimum vertex cover from a maximum matching. See J.
E. Hopcroft and R. M. Karp, "An :math:`n^{5/2}` Algorithm for Maximum
Matchings in Bipartite Graphs," SIAM Journal on Computing 2(4) (1973),
225-231, and D. Kőnig, "Gráfok és mátrixok," Matematikai és Fizikai
Lapok 38 (1931), 116-119.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching

from mathematicskit.graph_theory.core.base import BipartiteMatchingResult, Graph

__all__ = ["bipartite_matching"]


def bipartite_matching(graph: Graph, left) -> BipartiteMatchingResult:
    r"""A maximum matching of a bipartite graph, and a minimum vertex cover of the same size.

    König's construction: from every unmatched left vertex, follow
    alternating paths (non-matching edges left to right, matching edges
    right to left). With :math:`Z` the vertices reached, the cover is
    (left not in :math:`Z`) plus (right in :math:`Z`).

    Parameters
    ----------
    graph : Graph
        Undirected and bipartite.
    left : iterable of int
        The vertices on one side; every edge must join ``left`` to the rest.

    Returns
    -------
    BipartiteMatchingResult

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import complete_bipartite_graph
    >>> result = bipartite_matching(complete_bipartite_graph(2, 3), left=[0, 1])
    >>> result.size, len(result.vertex_cover)
    (2, 2)
    """
    left = sorted(left)
    left_set = set(left)
    right = [v for v in range(graph.n_vertices) if v not in left_set]
    row = {u: k for k, u in enumerate(left)}
    col = {v: k for k, v in enumerate(right)}
    rows, cols = [], []
    for u, v, _ in graph.edges():
        if (u in left_set) == (v in left_set):
            raise ValueError(f"edge ({u}, {v}) does not cross the bipartition")
        a, b = (u, v) if u in left_set else (v, u)
        rows.append(row[a])
        cols.append(col[b])
    biadjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(left), len(right)))
    match = maximum_bipartite_matching(biadjacency, perm_type="column")
    pairs = {left[i]: right[j] for i, j in enumerate(match) if j >= 0}
    partner_of_right = {v: u for u, v in pairs.items()}

    reached = set()
    queue = deque(u for u in left if u not in pairs)
    reached.update(queue)
    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if v in reached:
                continue
            reached.add(v)
            w = partner_of_right.get(v)
            if w is not None and w not in reached:
                reached.add(w)
                queue.append(w)
    cover = sorted([u for u in left if u not in reached] + [v for v in right if v in reached])
    return BipartiteMatchingResult(pairs=pairs, vertex_cover=cover)
