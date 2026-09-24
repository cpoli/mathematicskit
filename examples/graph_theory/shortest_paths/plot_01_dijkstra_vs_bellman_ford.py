r"""
Dijkstra's shortest-path algorithm (and when Bellman-Ford is needed)
==========================================================================

Runs Dijkstra's algorithm from one source on a small weighted graph and
draws the resulting shortest-path tree. Bellman-Ford gives the same
distances there, and takes over on a graph with a negative edge, where
Dijkstra's greedy assumption breaks down.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.graph_theory import Graph, bellman_ford_shortest_paths, dijkstra_shortest_paths
from mathematicskit.graph_theory.visualizers.plots import circular_layout, plot_graph

# %%
# Single-source shortest paths with Dijkstra
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
# The shortest-path tree
# -------------------------------------------------------------------------
#
# Each vertex's predecessor on its shortest path from vertex 0 defines a
# tree. The direct edge 0-1 (weight 4) is not used: the detour through
# vertex 2 costs only 3.

tree = [(int(p), v) for v, p in enumerate(dijkstra_result.predecessors) if p >= 0]
print("shortest-path tree edges:", tree)

pos = circular_layout(g.n_vertices)
fig, ax = plt.subplots(figsize=(5, 5))
plot_graph(g, ax=ax, positions=pos, highlight_edges=tree)
for u, v, w in g.edges():
    if u < v:
        ax.annotate(f"{w:g}", pos[[u, v]].mean(axis=0), ha="center", va="center", color="0.3", bbox={"fc": "white", "ec": "none"})
for v, d in enumerate(dijkstra_result.distances):
    ax.annotate(f"d = {d:g}", 1.3 * pos[v], ha="center", va="center", color="firebrick")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_title("Dijkstra's shortest-path tree from vertex 0")

# %%
# A directed graph with a negative edge: only Bellman-Ford applies
# -------------------------------------------------------------------------

g2 = Graph(3, directed=True)
g2.add_edge(0, 1, 4.0)
g2.add_edge(0, 2, 5.0)
g2.add_edge(1, 2, -2.0)
result = bellman_ford_shortest_paths(g2, sources=0)
print("\nBellman-Ford with a negative edge:", result.distances)
