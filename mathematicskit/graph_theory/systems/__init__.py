"""Concrete graph algorithms."""

from mathematicskit.graph_theory.systems.coloring import backtracking_coloring, greedy_coloring
from mathematicskit.graph_theory.systems.max_flow import max_flow_min_cut
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
]
