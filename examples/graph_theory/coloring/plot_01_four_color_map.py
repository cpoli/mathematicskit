r"""
The four color theorem: coloring a planar map
===================================================

Draws a map of 20 countries, turns it into a planar graph (one vertex
per country, an edge for each shared border), and colors it. Exact
backtracking search needs exactly four colors, as Guthrie conjectured
and Appel and Haken proved, while the fast greedy heuristic can waste a
fifth color on the same map.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from mathematicskit.graph_theory import Graph, backtracking_coloring, complete_graph, greedy_coloring

# %%
# A map of countries
# -----------------------------------------------------
#
# Each pixel belongs to the country whose capital is nearest. A central
# country ringed by five neighbours -- an odd ring -- already forces a
# fourth color, since the ring alone needs three and the center touches
# all of them.

rng = np.random.default_rng(3)
ring = [(0.5 + 0.2 * np.cos(t), 0.5 + 0.2 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
outer = rng.uniform(0.0, 1.0, size=(60, 2))
outer = outer[np.hypot(*(outer - 0.5).T) > 0.35][:14]
capitals = np.vstack([[0.5, 0.5], ring, outer])

size = 300
yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
labels = np.argmin((xx[..., None] - capitals[:, 0]) ** 2 + (yy[..., None] - capitals[:, 1]) ** 2, axis=-1)

# %%
# Countries sharing a border become adjacent vertices
# -----------------------------------------------------

g = Graph(len(capitals))
for a, b in zip(np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()]), np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])):
    if a != b:
        g.add_edge(int(a), int(b))
print(f"{g.n_vertices} countries, {len(g.edges())} shared borders")

optimal = backtracking_coloring(g)
proper = all(optimal.coloring[u] != optimal.coloring[v] for u, v, _ in g.edges())
print(f"exact backtracking: {optimal.num_colors} colors (proper coloring: {proper})")

# %%
# The greedy heuristic can need more than four
# -----------------------------------------------------

worst = max((greedy_coloring(g, order=rng.permutation(g.n_vertices)) for _ in range(300)), key=lambda r: r.num_colors)
print(f"greedy, worst of 300 vertex orders: {worst.num_colors} colors")

# %%
# Planarity matters: the complete graph K5, which no map can realise,
# needs five colors.

print(f"K5 (not planar): {backtracking_coloring(complete_graph(5)).num_colors} colors")

# %%
# The colored maps
# -----------------------------------------------------

palette = ListedColormap(["#e15759", "#4e79a7", "#59a14f", "#edc948", "#b07aa1", "#9c755f", "#76b7b2"])
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
for ax, result, title in [(axes[0], optimal, "exact search"), (axes[1], worst, "greedy heuristic")]:
    image = np.vectorize(result.coloring.get)(labels)
    ax.imshow(image, cmap=palette, vmin=0, vmax=palette.N - 1, origin="lower", interpolation="nearest")
    border = (np.diff(labels, axis=0, prepend=labels[:1]) != 0) | (np.diff(labels, axis=1, prepend=labels[:, :1]) != 0)
    ax.imshow(np.ma.masked_where(~border, border), cmap=ListedColormap(["black"]), origin="lower", interpolation="nearest")
    ax.plot(*(capitals * (size - 1)).T, "k.", ms=4)
    ax.set_title(f"{title}: {result.num_colors} colors")
    ax.axis("off")
fig.tight_layout()
