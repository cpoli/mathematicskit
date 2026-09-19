"""mathkit.graph_theory: graph algorithms on a lightweight own graph container.

A minimal ``Graph`` container (adjacency list, no ``networkx``
dependency) supports: shortest-path algorithms
(``scipy.sparse.csgraph.dijkstra``/``bellman_ford``/``floyd_warshall``);
minimum spanning tree (``scipy.sparse.csgraph.minimum_spanning_tree``,
Kruskal-based, alongside a hand-rolled Prim's implementation kept for
comparison); maximum flow / minimum cut
(``scipy.sparse.csgraph.maximum_flow``); graph coloring (greedy
heuristic and exact backtracking, hand-rolled -- NP-complete in
general, no scipy equivalent); and spectral graph theory (graph
Laplacian via ``scipy.sparse.csgraph.laplacian``, algebraic connectivity
and spectral bipartition via ``numpy.linalg.eigh``).
"""

from mathkit.graph_theory.core.base import ColoringResult, Graph, MaxFlowResult, MSTResult, ShortestPathResult, SpectralResult
from mathkit.graph_theory.systems.coloring import backtracking_coloring, greedy_coloring
from mathkit.graph_theory.systems.max_flow import max_flow_min_cut
from mathkit.graph_theory.systems.shortest_paths import bellman_ford_shortest_paths, dijkstra_shortest_paths, floyd_warshall_shortest_paths
from mathkit.graph_theory.systems.spanning_tree import kruskal_mst, prim_mst
from mathkit.graph_theory.systems.spectral import spectral_analysis
from mathkit.graph_theory.utils.generators import complete_graph, cycle_graph, path_graph, random_graph

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Graph",
    "ShortestPathResult",
    "MSTResult",
    "MaxFlowResult",
    "ColoringResult",
    "SpectralResult",
    "dijkstra_shortest_paths",
    "bellman_ford_shortest_paths",
    "floyd_warshall_shortest_paths",
    "kruskal_mst",
    "prim_mst",
    "max_flow_min_cut",
    "greedy_coloring",
    "backtracking_coloring",
    "spectral_analysis",
    "complete_graph",
    "cycle_graph",
    "path_graph",
    "random_graph",
]
