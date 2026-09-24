r"""Connected components, and the giant component of random graphs.

Built on :func:`scipy.sparse.csgraph.connected_components`. See P.
Erdős and A. Rényi, "On the Evolution of Random Graphs," Publications of
the Mathematical Institute of the Hungarian Academy of Sciences 5
(1960), 17-61.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components as _scipy_components

from mathematicskit.graph_theory.core.base import ComponentsResult, Graph

__all__ = ["connected_components", "giant_component_fraction"]


def connected_components(graph: Graph) -> ComponentsResult:
    r"""The connected components of an undirected graph (weak components if directed).

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    ComponentsResult

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import path_graph
    >>> connected_components(path_graph(4)).n_components
    1
    """
    n, labels = _scipy_components(graph.to_sparse(), directed=graph.directed, connection="weak")
    sizes = np.sort(np.bincount(labels, minlength=n))[::-1]
    return ComponentsResult(n_components=int(n), labels=labels, sizes=sizes)


def giant_component_fraction(graph: Graph) -> float:
    r"""Fraction of the vertices in the largest connected component.

    In the Erdős-Rényi graph :math:`G(n, c/n)` this fraction tends to 0
    for :math:`c < 1` and to the positive root :math:`s` of
    :math:`s = 1 - e^{-cs}` for :math:`c > 1`.

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    float

    Examples
    --------
    >>> from mathematicskit.graph_theory.utils.generators import complete_graph
    >>> giant_component_fraction(complete_graph(5))
    1.0
    """
    return float(connected_components(graph).sizes[0] / graph.n_vertices)
