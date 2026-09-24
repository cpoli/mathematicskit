r"""
Barnsley's fern: an iterated function system
==================================================

Barnsley's fern is the attractor of just four contractive affine maps,
drawn with the chaos game: start anywhere, and at each step apply one
of the maps chosen at random with Barnsley's probabilities. Hutchinson's
theorem says the fern is the unique set equal to the union of its four
images under the maps, which the second figure shows directly.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import BarnsleyFern
from mathematicskit.fractals_chaos.visualizers.plots import plot_ifs_points

# %%
# The fern from the chaos game
# ----------------------------

fern = BarnsleyFern()
points = fern.generate(60000, seed=0)
ax = plot_ifs_points(points, color="darkgreen")
ax.set_title("Barnsley fern (60 000 chaos-game points)")

for name, (A, _, p) in zip(["stem", "main copy", "left leaflet", "right leaflet"], fern.transforms):
    print(f"{name:14s} p = {p:.2f}, contraction |det A| = {abs(np.linalg.det(A)):.3f}")

# %%
# The fern is four smaller ferns
# ------------------------------
#
# Applying each map to the whole fern gives four pieces -- the stem, the
# fern minus its lowest leaflets, and the two lowest leaflets -- whose
# union is the fern again.

fig, ax = plt.subplots(figsize=(5, 8))
colors = ["saddlebrown", "darkgreen", "royalblue", "darkorange"]
for (A, b, _), color in zip(fern.transforms, colors):
    image = points @ A.T + b
    ax.scatter(image[:, 0], image[:, 1], s=0.2, color=color, linewidths=0)
ax.set_aspect("equal")
ax.set_title("Images of the fern under its four maps")
