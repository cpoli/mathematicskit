r"""
Klein's Erlangen program: the symmetry group of a square
==============================================================

Builds the dihedral group D_4 of the eight symmetries of a square,
checks its defining relations, and draws its Cayley table.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.abstract_algebra import DihedralGroup
from mathematicskit.abstract_algebra.visualizers.plots import plot_cayley_table

# %%
# Rotations and reflections of the square
# -----------------------------------------------------

d4 = DihedralGroup(4)
r, s = d4.rotation, d4.reflection
print(f"|D_4| = {d4.order}, abelian: {d4.is_abelian()}")
print(f"rotation r = {r}, order {d4.element_order(r)}")
print(f"reflection s = {s}, order {d4.element_order(s)}")
print(f"s r s = r^-1: {d4.operate(d4.operate(s, r), s) == d4.inverse(r)}")

# %%
# Where each symmetry sends the labelled corners
# -----------------------------------------------------

corners = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, g in zip(axes.flat, d4.elements):
    ax.fill(*corners.T, color="0.9", edgecolor="0.3")
    for vertex, image in enumerate(g):
        ax.annotate(str(vertex), corners[image] * 1.2, ha="center", va="center")
    ax.set_title(str(g), fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle("The 8 symmetries of a square (vertex labels after each move)")

# %%
# Cayley table
# -----------------------------------------------------

plot_cayley_table(d4)
