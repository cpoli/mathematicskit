r"""
Cayley's formula via Prüfer codes
=======================================

Encodes labeled trees as Prüfer sequences and decodes every sequence of
length n-2 back to a tree. The correspondence is one-to-one, so there
are n^(n-2) labeled trees on n vertices.
"""

# %%
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import count_labeled_trees, prufer_decode, prufer_encode

# %%
# Encode and decode one tree
# -----------------------------------------------------

tree = [(0, 4), (1, 4), (2, 5), (3, 5), (4, 5), (5, 6)]
code = prufer_encode(tree, 7)
print(f"tree {tree}\nPrüfer code {code}\ndecoded {prufer_decode(code)}")

# %%
# All 16 labeled trees on 4 vertices
# -----------------------------------------------------

n = 4
positions = {v: (np.cos(2 * np.pi * v / n), np.sin(2 * np.pi * v / n)) for v in range(n)}
sequences = list(product(range(n), repeat=n - 2))
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax, seq in zip(axes.flat, sequences):
    for u, v in prufer_decode(seq):
        ax.plot(*zip(positions[u], positions[v]), "k-")
    for v, (x, y) in positions.items():
        ax.text(x, y, str(v), ha="center", va="center", bbox={"boxstyle": "circle", "fc": "w"})
    ax.set_title(str(list(seq)), fontsize=9)
    ax.axis("off")
fig.suptitle(f"{len(sequences)} codes = {count_labeled_trees(n)} = 4^2 labeled trees")

# %%
# Counts for small n
# -----------------------------------------------------

for n in range(2, 8):
    decoded = {tuple(prufer_decode(s)) for s in product(range(n), repeat=n - 2)}
    print(f"n = {n}: {len(decoded)} distinct trees, n^(n-2) = {count_labeled_trees(n)}")
