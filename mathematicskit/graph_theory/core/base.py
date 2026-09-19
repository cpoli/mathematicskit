r"""A lightweight own graph container, and result dataclasses shared
across mathematicskit.graph_theory's systems/ modules.

:class:`Graph` is deliberately minimal (adjacency-list storage, no
``networkx`` dependency) -- its job is to hold vertices/edges and hand
off a :func:`scipy.sparse.csr_matrix` representation to whichever
``scipy.sparse.csgraph`` routine a ``systems/`` function needs, per
mathematicskit's "call the library directly" rule for anything scipy already
implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse

__all__ = ["Graph", "ShortestPathResult", "MSTResult", "MaxFlowResult", "ColoringResult", "SpectralResult"]


class Graph:
    r"""A lightweight directed or undirected weighted graph.

    Vertices are the integers ``0, ..., n_vertices - 1``. Stored as an
    adjacency dict of dicts (``self._adj[u][v] = weight``); edges with no
    explicit weight default to 1.0. Deliberately has no algorithms of its
    own beyond construction/conversion -- every algorithm lives in
    ``systems/``, most calling :func:`to_sparse` to hand a
    :class:`scipy.sparse.csr_matrix` to ``scipy.sparse.csgraph``.

    Parameters
    ----------
    n_vertices : int
    directed : bool

    Examples
    --------
    >>> g = Graph(3)
    >>> g.add_edge(0, 1, weight=2.0)
    >>> g.add_edge(1, 2, weight=3.0)
    >>> g.to_sparse().toarray()
    array([[0., 2., 0.],
           [2., 0., 3.],
           [0., 3., 0.]])
    """

    def __init__(self, n_vertices: int, directed: bool = False):
        self.n_vertices = int(n_vertices)
        self.directed = directed
        self._adj: dict = {i: {} for i in range(self.n_vertices)}

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Add an edge (or update its weight if it already exists).

        Parameters
        ----------
        u, v : int
        weight : float
        """
        self._adj[u][v] = float(weight)
        if not self.directed:
            self._adj[v][u] = float(weight)

    def neighbors(self, u: int) -> dict:
        """dict: ``{neighbor: weight}`` for vertex ``u``."""
        return self._adj[u]

    def edges(self) -> list:
        """list of (int, int, float): Every edge as ``(u, v, weight)``.

        For an undirected graph, each edge is listed once (``u < v``).
        """
        result = []
        for u, neighbors in self._adj.items():
            for v, w in neighbors.items():
                if self.directed or u <= v:
                    result.append((u, v, w))
        return result

    def to_sparse(self):
        """Build a :class:`scipy.sparse.csr_matrix` adjacency/weight matrix.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_vertices, n_vertices)
        """
        rows, cols, data = [], [], []
        for u, neighbors in self._adj.items():
            for v, w in neighbors.items():
                rows.append(u)
                cols.append(v)
                data.append(w)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n_vertices, self.n_vertices))


@dataclass
class ShortestPathResult:
    """Container for a shortest-path computation."""

    distances: np.ndarray
    """ndarray: Shortest-path distance(s); shape (n,) for a single
    source, or (n_sources, n) for multiple sources."""

    predecessors: np.ndarray
    """ndarray, int: Predecessor-tree indices (``-9999`` for
    unreachable/root, following ``scipy.sparse.csgraph``'s convention)."""

    method: str = ""
    """str: e.g. ``"dijkstra"``, ``"bellman_ford"``, ``"floyd_warshall"``."""


@dataclass
class MSTResult:
    """Container for a minimum-spanning-tree/-forest computation."""

    edges: list
    """list of (int, int, float): The MST's edges as ``(u, v, weight)``."""

    total_weight: float
    method: str = ""
    """str: e.g. ``"kruskal"`` (via scipy) or ``"prim"`` (hand-rolled)."""


@dataclass
class MaxFlowResult:
    """Container for a maximum-flow computation."""

    flow_value: float
    flow_matrix: np.ndarray
    """ndarray, shape (n, n): Flow assigned to each edge."""

    min_cut: Optional[tuple] = None
    """tuple of (ndarray, ndarray), optional: ``(source_side, sink_side)``
    vertex partitions defining the corresponding minimum cut."""


@dataclass
class ColoringResult:
    """Container for a graph-coloring result."""

    coloring: dict
    """dict: ``{vertex: color_index}``."""

    num_colors: int
    method: str = ""
    """str: ``"greedy"`` or ``"backtracking"``."""


@dataclass
class SpectralResult:
    """Container for a spectral graph-theory computation."""

    laplacian: np.ndarray
    eigenvalues: np.ndarray
    """ndarray, shape (n,): Laplacian eigenvalues, ascending."""

    eigenvectors: np.ndarray
    algebraic_connectivity: float
    """float: The second-smallest Laplacian eigenvalue (Fiedler value);
    zero iff the graph is disconnected."""

    bipartition: np.ndarray = field(default_factory=lambda: np.array([]))
    """ndarray, bool: Spectral bipartition from the Fiedler vector's sign
    (``True``/``False`` per vertex)."""
