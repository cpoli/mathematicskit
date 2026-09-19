r"""Spectral graph theory: the graph Laplacian, algebraic connectivity,
and spectral bipartition.

The Laplacian itself is built via :func:`scipy.sparse.csgraph.laplacian`
and its eigendecomposition via :func:`numpy.linalg.eigh` (the Laplacian
is symmetric for an undirected graph) -- mathkit does not reimplement
either. See Chung, *Spectral Graph Theory*, 1997, Ch. 1, and Fiedler
(1973), *Algebraic connectivity of graphs*, Czechoslovak Math. J. 23.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csgraph

from mathkit.graph_theory.core.base import Graph, SpectralResult

__all__ = ["spectral_analysis"]


def spectral_analysis(graph: Graph) -> SpectralResult:
    r"""Graph Laplacian, its spectrum, algebraic connectivity, and spectral bipartition.

    The (combinatorial) Laplacian :math:`L = D - A` (degree matrix minus
    adjacency), via :func:`scipy.sparse.csgraph.laplacian`, is symmetric
    positive semi-definite for an undirected graph, with smallest
    eigenvalue always 0 (eigenvector: all-ones). The second-smallest
    eigenvalue (the *algebraic connectivity*, or Fiedler value) is
    positive iff the graph is connected, and larger values indicate a
    more robustly connected graph; its eigenvector (the Fiedler vector)
    partitions vertices by sign into two well-separated communities --
    spectral clustering's simplest form. See Chung, *Spectral Graph
    Theory*, 1997, Ch. 1, and Fiedler (1973), Czechoslovak Math. J. 23.

    Parameters
    ----------
    graph : Graph
        Undirected.

    Returns
    -------
    SpectralResult

    Examples
    --------
    >>> g = Graph(4)
    >>> g.add_edge(0, 1)
    >>> g.add_edge(1, 2)
    >>> g.add_edge(2, 3)
    >>> g.add_edge(3, 0)
    >>> result = spectral_analysis(g)
    >>> np.round(result.eigenvalues, 6)
    array([-0.,  2.,  2.,  4.])
    >>> round(result.algebraic_connectivity, 6)
    2.0
    """
    laplacian = csgraph.laplacian(graph.to_sparse()).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    algebraic_connectivity = float(eigenvalues[1]) if eigenvalues.shape[0] > 1 else 0.0
    fiedler_vector = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(graph.n_vertices)
    bipartition = fiedler_vector >= 0.0
    return SpectralResult(
        laplacian=laplacian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        algebraic_connectivity=algebraic_connectivity,
        bipartition=bipartition,
    )
