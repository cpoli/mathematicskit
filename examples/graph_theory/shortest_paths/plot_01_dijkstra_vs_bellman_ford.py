r"""
Dijkstra vs. Bellman-Ford, and all-pairs Floyd-Warshall
==============================================================

Compares the three shortest-path algorithms on the same weighted graph,
including a negative-weight edge that only Bellman-Ford can handle
correctly.
"""

# %%
from mathematicskit.graph_theory import Graph, bellman_ford_shortest_paths, dijkstra_shortest_paths, floyd_warshall_shortest_paths

# %%
# A graph with non-negative weights: Dijkstra and Bellman-Ford agree
# -------------------------------------------------------------------------

g = Graph(5)
g.add_edge(0, 1, 4.0)
g.add_edge(0, 2, 1.0)
g.add_edge(2, 1, 2.0)
g.add_edge(1, 3, 1.0)
g.add_edge(2, 3, 5.0)
g.add_edge(3, 4, 3.0)

dijkstra_result = dijkstra_shortest_paths(g, sources=0)
bf_result = bellman_ford_shortest_paths(g, sources=0)
print("Dijkstra distances from 0:     ", dijkstra_result.distances)
print("Bellman-Ford distances from 0: ", bf_result.distances)

# %%
# A directed graph with a negative edge: only Bellman-Ford applies
# -------------------------------------------------------------------------

g2 = Graph(3, directed=True)
g2.add_edge(0, 1, 4.0)
g2.add_edge(0, 2, 5.0)
g2.add_edge(1, 2, -2.0)
result = bellman_ford_shortest_paths(g2, sources=0)
print("\nBellman-Ford with a negative edge:", result.distances)

# %%
# All-pairs distances via Floyd-Warshall
# -----------------------------------------------------

fw_result = floyd_warshall_shortest_paths(g)
print("\nall-pairs distance matrix:")
print(fw_result.distances)
