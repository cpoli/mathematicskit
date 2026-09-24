r"""
A* search on a grid with obstacles
========================================

Finds a shortest path across a grid with a wall. With the Manhattan
distance as its heuristic, A* reaches the same distance as Dijkstra's
algorithm while expanding far fewer vertices.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import astar_shortest_path, grid_graph

# %%
# Build the grid and search it twice
# -----------------------------------------------------

rows = cols = 30
blocked = [(r, 15) for r in range(0, 24)] + [(8, c) for c in range(3, 15)]
g, pos = grid_graph(rows, cols, blocked=blocked)
source, target = 2 * cols + 2, (rows - 3) * cols + (cols - 3)


def manhattan(v):
    return float(np.abs(pos[v] - pos[target]).sum())


guided = astar_shortest_path(g, source, target, manhattan)
blind = astar_shortest_path(g, source, target)
print(f"A* (Manhattan): distance {guided.distance:.0f}, expanded {guided.n_expanded} vertices")
print(f"Dijkstra (h = 0): distance {blind.distance:.0f}, expanded {blind.n_expanded} vertices")

fig, ax = plt.subplots(figsize=(6, 6))
wall = np.array([(c, r) for r, c in blocked])
ax.plot(*wall.T, "ks", ms=6)
ax.plot(*pos[guided.path].T, "-", color="tab:red", lw=2.5, label="A* path")
ax.plot(*pos[[source, target]].T, "o", color="tab:green", ms=10)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.legend()
