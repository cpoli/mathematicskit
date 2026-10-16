r"""
Vietoris's complex of a point cloud (1927)
==========================================

To define homology for any compact metric space, Leopold Vietoris
joined every set of points that are pairwise close. Today this is the
Vietoris-Rips complex: at scale :math:`r` (balls of radius :math:`r`),
a simplex for every set of points at most :math:`2r` apart. Sampled from
a figure eight, the complex at small scales is dust. At the right
scale it has :math:`\beta = (1, 2)`, the figure eight's two loops. At
large scales everything fills in. Triangles are enough to count loops,
and mod-2 coefficients keep the rank computation fast.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import betti_numbers, vietoris_rips_complex
from mathematicskit.topology.visualizers import plot_complex

rng = np.random.default_rng(4)
t = rng.uniform(0, 2 * np.pi, 80)
points = np.column_stack([np.sin(t), np.sin(t) * np.cos(t)]) * [2, 2] + rng.normal(0, 0.04, (80, 2))

radii = [0.05, 0.15, 0.3, 0.6]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, r in zip(axes, radii, strict=True):
    K = vietoris_rips_complex(points, r, max_dim=2)
    betti = betti_numbers(K, field=2)[:2]
    print(f"r = {r:4.2f}: f = {K.f_vector}, (beta_0, beta_1) = {betti}")
    plot_complex(K, ax=ax, alpha=0.25, vertex_size=6)
    ax.set_title(f"r = {r}: $(\\beta_0, \\beta_1)$ = {betti}")
    ax.axis("off")
fig.suptitle("Vietoris-Rips complexes of a noisy figure eight")
