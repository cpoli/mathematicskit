r"""
Hierholzer's algorithm: constructing an Eulerian circuit
========================================================

Euler showed that a closed walk using every edge once forces every
vertex to have even degree. Hierholzer (1873) proved the converse for a
connected graph, and his proof is an algorithm: walk along unused edges
until you are stuck, which can only happen back where you started;
then, from any vertex of that closed walk that still has unused edges,
walk a second closed tour and splice it in. Repeat until no edge is
left.

The octahedron below has every vertex of degree 4. A greedy walk from
vertex 0 gets stuck back at 0 after 9 of its 12 edges, and the
remaining triangle has to be spliced in.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import Graph, eulerian_circuit, odd_degree_vertices

opposite = {(0, 3), (1, 4), (2, 5)}
octahedron = Graph(6)
for u in range(6):
    for v in range(u + 1, 6):
        if (u, v) not in opposite:
            octahedron.add_edge(u, v)
print(f"edges: {len(octahedron.edges())}, odd-degree vertices: {odd_degree_vertices(octahedron)}")

# %%
# Getting stuck, then splicing
# ------------------------------
#
# Walking greedily from vertex 0 (always to the lowest-numbered unused
# neighbour) ends back at 0 with three edges unused. Hierholzer's insight
# is that this is not a failure: the unused edges still have even degree
# at every vertex, so they form closed tours, and each one starts at a
# vertex already on the walk, where it can be inserted.


def greedy_closed_walk(graph, start, used):
    walk = [start]
    while True:
        options = [v for v in sorted(graph.neighbors(walk[-1])) if frozenset((walk[-1], v)) not in used]
        if not options:
            return walk
        used.add(frozenset((walk[-1], options[0])))
        walk.append(options[0])


used: set = set()
tours = [greedy_closed_walk(octahedron, 0, used)]
while len(used) < len(octahedron.edges()):
    start = next(v for tour in tours for v in tour if any(frozenset((v, w)) not in used for w in octahedron.neighbors(v)))
    tours.append(greedy_closed_walk(octahedron, start, used))
for k, tour in enumerate(tours):
    print(f"closed tour {k + 1}: {tour}")

circuit = eulerian_circuit(octahedron, start=0)
print(f"eulerian_circuit: {circuit} ({len(circuit) - 1} edges)")

# %%
# Drawing the tours and the circuit
# -----------------------------------

angles = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
outer = np.column_stack([np.cos(angles), np.sin(angles)])
pos = np.vstack([outer, -0.45 * outer])  # vertex k + 3 sits opposite vertex k


def draw_edges(ax, walk, color, numbered=False):
    for k, (u, v) in enumerate(zip(walk, walk[1:], strict=False)):
        p, q = pos[u], pos[v]
        ax.annotate("", q, p, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2, "shrinkA": 12, "shrinkB": 12})
        if numbered:
            ax.annotate(str(k + 1), (p + q) / 2, ha="center", va="center", fontsize=8, bbox={"boxstyle": "circle", "fc": "white", "ec": color})


fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
colors = plt.cm.tab10.colors
for ax in axes:
    for u, v, _ in octahedron.edges():
        ax.plot(*pos[[u, v]].T, color="0.85", lw=4, zorder=0)
    ax.scatter(*pos.T, s=500, color="white", edgecolors="black", zorder=3)
    for v, (x, y) in enumerate(pos):
        ax.annotate(str(v), (x, y), ha="center", va="center", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
for k, tour in enumerate(tours):
    draw_edges(axes[0], tour, colors[k])
axes[0].set_title(f"Greedy walking leaves {len(tours)} closed tours")
draw_edges(axes[1], circuit, "tab:blue", numbered=True)
axes[1].set_title("Spliced together: one Eulerian circuit")
fig.tight_layout()

# %%
# An application: de Bruijn sequences
# -------------------------------------
#
# A cyclic binary string in which every 4-bit word appears exactly once
# is an Eulerian circuit of the de Bruijn graph on 3-bit words, whose
# edges append one bit. Every vertex there has in-degree equal to
# out-degree 2.

de_bruijn = Graph(8, directed=True)
for word in range(8):
    for bit in (0, 1):
        de_bruijn.add_edge(word, ((word << 1) | bit) & 7)
sequence = "".join(str(w & 1) for w in eulerian_circuit(de_bruijn)[1:])
words = sorted({(sequence + sequence[:3])[i : i + 4] for i in range(len(sequence))})
print(f"de Bruijn sequence: {sequence} contains {len(words)} distinct 4-bit words")

plt.show()
