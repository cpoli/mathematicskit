r"""
Floyd-Warshall: all-pairs shortest paths
==============================================

Computes every shortest-path distance in a weighted graph at once with
the Floyd-Warshall algorithm, and checks that it agrees with running
Dijkstra's algorithm from every vertex.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import Graph, dijkstra_shortest_paths, floyd_warshall_shortest_paths

# %%
# A small road network
# -----------------------------------------------------

rng = np.random.default_rng(1)
n = 12
g = Graph(n)
for u in range(n):
    for v in range(u + 1, n):
        if rng.random() < 0.3:
            g.add_edge(u, v, weight=float(rng.integers(1, 10)))

all_pairs = floyd_warshall_shortest_paths(g).distances
from_each = dijkstra_shortest_paths(g).distances
print(f"Floyd-Warshall agrees with n Dijkstra runs: {np.allclose(all_pairs, from_each)}")
print(f"diameter (longest shortest path): {np.max(all_pairs[np.isfinite(all_pairs)]):.0f}")

fig, ax = plt.subplots()
image = ax.imshow(all_pairs, cmap="viridis")
fig.colorbar(image, ax=ax, label="shortest-path distance")
ax.set_title("All-pairs distance matrix")
