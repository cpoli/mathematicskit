r"""Counting spanning trees with Kirchhoff's matrix-tree theorem.

The count is a determinant of the reduced Laplacian, computed with
:func:`numpy.linalg.slogdet` for numerical range. See G. Kirchhoff,
"Ueber die Auflösung der Gleichungen, auf welche man bei der
Untersuchung der linearen Vertheilung galvanischer Ströme geführt wird,"
Annalen der Physik und Chemie 72(12) (1847), 497-508.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.graph_theory.core.base import Graph

__all__ = ["count_spanning_trees"]


def count_spanning_trees(graph: Graph) -> int:
    r"""The number of spanning trees of an undirected graph, :math:`\det L_{(0)}`.

    Deleting any one row and the matching column of the Laplacian
    :math:`L = D - A` leaves a matrix whose determinant counts spanning
    trees (weighted by the product of edge weights, when edges are
    weighted). Exact for integer counts up to about :math:`10^{15}`.

    Parameters
    ----------
    graph : Graph
        Undirected; multigraph-free.

    Returns
    -------
    int

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import complete_graph
    >>> count_spanning_trees(complete_graph(5))  # Cayley: 5^3
    125
    """
    if graph.directed:
        raise ValueError("the matrix-tree theorem here is for undirected graphs")
    adjacency = graph.to_sparse().toarray()
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    if graph.n_vertices == 1:
        return 1
    sign, logdet = np.linalg.slogdet(laplacian[1:, 1:])
    return 0 if sign <= 0 else int(round(np.exp(logdet)))
