"""mathematicskit.graph_theory: graph algorithms on a lightweight own graph container.

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
and spectral bipartition via ``numpy.linalg.eigh``); Kirchhoff's
spanning-tree count; Hamiltonian cycles by backtracking; bipartite
matching (Hopcroft-Karp via scipy) with König vertex covers; Turán
graphs and clique numbers; the assignment problem
(``scipy.optimize.linear_sum_assignment``); connected components and
the Erdős-Rényi giant component; A* search; and PageRank.
"""

from mathematicskit.graph_theory.core.base import (
    AssignmentResult,
    BipartiteMatchingResult,
    ColoringResult,
    ComponentsResult,
    Graph,
    MaxFlowResult,
    MSTResult,
    PageRankResult,
    SearchResult,
    ShortestPathResult,
    SpectralResult,
)
from mathematicskit.graph_theory.systems.assignment import solve_assignment
from mathematicskit.graph_theory.systems.coloring import backtracking_coloring, greedy_coloring
from mathematicskit.graph_theory.systems.components import connected_components, giant_component_fraction
from mathematicskit.graph_theory.systems.enumeration import count_spanning_trees
from mathematicskit.graph_theory.systems.extremal import clique_number, turan_graph, turan_number
from mathematicskit.graph_theory.systems.hamiltonian import hamiltonian_cycle
from mathematicskit.graph_theory.systems.matching import bipartite_matching
from mathematicskit.graph_theory.systems.max_flow import max_flow_min_cut
from mathematicskit.graph_theory.systems.ranking import pagerank
from mathematicskit.graph_theory.systems.search import astar_shortest_path
from mathematicskit.graph_theory.systems.shortest_paths import bellman_ford_shortest_paths, dijkstra_shortest_paths, floyd_warshall_shortest_paths
from mathematicskit.graph_theory.systems.spanning_tree import kruskal_mst, prim_mst
from mathematicskit.graph_theory.systems.spectral import spectral_analysis
from mathematicskit.graph_theory.utils.generators import (
    complete_bipartite_graph,
    complete_graph,
    cycle_graph,
    dodecahedron_graph,
    grid_graph,
    path_graph,
    random_graph,
)

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
    "BipartiteMatchingResult",
    "AssignmentResult",
    "ComponentsResult",
    "SearchResult",
    "PageRankResult",
    "dodecahedron_graph",
    "grid_graph",
    "complete_bipartite_graph",
    "solve_assignment",
    "connected_components",
    "giant_component_fraction",
    "count_spanning_trees",
    "clique_number",
    "turan_graph",
    "turan_number",
    "hamiltonian_cycle",
    "bipartite_matching",
    "pagerank",
    "astar_shortest_path",
]
