r"""PageRank by power iteration.

Uses sparse matrix-vector products from :mod:`scipy.sparse`; the
iteration itself is the subject, so it is written out rather than
delegated to an eigensolver. See S. Brin and L. Page, "The Anatomy of a
Large-Scale Hypertextual Web Search Engine," Computer Networks and ISDN
Systems 30(1-7) (1998), 107-117.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER
from mathematicskit.graph_theory.core.base import Graph, PageRankResult

__all__ = ["pagerank"]


def pagerank(graph: Graph, damping: float = 0.85, tol: float = 1e-12, max_iter: int = DEFAULT_MAX_ITER) -> PageRankResult:
    r"""The PageRank vector: the stationary distribution of a random surfer.

    With probability ``damping`` the surfer follows a random out-link
    (edge weights act as relative link strengths); otherwise, or from a
    page with no out-links, it jumps to a uniformly random page. The
    scores solve

    .. math::

       \pi = d\, \pi P + \frac{1-d}{n}\mathbf{1},

    and power iteration converges at rate ``damping``.

    Parameters
    ----------
    graph : Graph
        Directed (an undirected edge counts as links both ways).
    damping : float
    tol : float
        Stop when the L1 change between iterates falls below ``tol``.
    max_iter : int

    Returns
    -------
    PageRankResult

    Examples
    --------
    >>> from mathematicskit.graph_theory.core.base import Graph
    >>> g = Graph(3, directed=True)
    >>> for u, v in [(0, 1), (1, 2), (2, 0), (0, 2)]:
    ...     g.add_edge(u, v)
    >>> np.round(pagerank(g).scores, 4).tolist()
    [0.3878, 0.2148, 0.3974]
    """
    n = graph.n_vertices
    weights = graph.to_sparse().tocsr()
    out = np.asarray(weights.sum(axis=1)).ravel()
    dangling = out == 0
    inv_out = np.divide(1.0, out, out=np.zeros_like(out), where=~dangling)
    transition_t = (weights.multiply(inv_out[:, None])).T.tocsr()
    scores = np.full(n, 1.0 / n)
    for iteration in range(1, max_iter + 1):
        new = damping * (transition_t @ scores + scores[dangling].sum() / n) + (1 - damping) / n
        if np.abs(new - scores).sum() < tol:
            return PageRankResult(scores=new, iterations=iteration)
        scores = new
    return PageRankResult(scores=scores, iterations=max_iter)
