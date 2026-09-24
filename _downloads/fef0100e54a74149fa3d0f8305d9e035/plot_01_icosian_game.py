r"""
Hamilton's icosian game
=============================

Solves Hamilton's 1857 puzzle: travel along the edges of a dodecahedron
through all 20 vertices exactly once and return to the start. The same
search shows that the Petersen graph has no such cycle.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import Graph, dodecahedron_graph, hamiltonian_cycle

# %%
# A tour of the dodecahedron
# -----------------------------------------------------

g = dodecahedron_graph()
tour = hamiltonian_cycle(g)
print(f"Hamiltonian cycle: {tour}")

radius = {0: 3.0, 1: 2.0, 2: 1.4, 3: 0.7}


def position(v):
    if v < 5:
        return radius[0], 2 * np.pi * v / 5
    if v < 15:
        k = v - 5
        return (radius[1] if k % 2 == 0 else radius[2]), 2 * np.pi * k / 10
    return radius[3], 2 * np.pi * (v - 15) / 5 + 2 * np.pi / 10


pos = np.array([[r * np.sin(t), r * np.cos(t)] for r, t in map(position, range(20))])
fig, ax = plt.subplots(figsize=(6, 6))
for u, v, _ in g.edges():
    ax.plot(*pos[[u, v]].T, color="0.8", lw=1)
cycle = tour + tour[:1]
ax.plot(*pos[cycle].T, color="tab:red", lw=2.5)
ax.plot(*pos.T, "ko")
for v, (x, y) in enumerate(pos):
    ax.annotate(str(v), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("The icosian game (Schlegel diagram of the dodecahedron)")

# %%
# The Petersen graph has no Hamiltonian cycle
# -----------------------------------------------------

petersen = Graph(10)
for i in range(5):
    petersen.add_edge(i, (i + 1) % 5)
    petersen.add_edge(i, i + 5)
    petersen.add_edge(5 + i, 5 + (i + 2) % 5)
print(f"Petersen graph: {hamiltonian_cycle(petersen)}")
