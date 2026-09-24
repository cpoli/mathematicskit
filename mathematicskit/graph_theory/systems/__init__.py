"""Concrete graph algorithms."""

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

__all__ = [
    "dijkstra_shortest_paths",
    "bellman_ford_shortest_paths",
    "floyd_warshall_shortest_paths",
    "kruskal_mst",
    "prim_mst",
    "max_flow_min_cut",
    "greedy_coloring",
    "backtracking_coloring",
    "spectral_analysis",
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
