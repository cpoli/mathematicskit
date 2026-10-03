r"""
Čech complexes and the nerve theorem (1932-1948)
================================================

Eduard Čech built homology from the *nerve* of a cover: a simplex for
every set of cover sets that share a point. Borsuk's nerve theorem
(1948) says that when every intersection is contractible, as for round
discs, the nerve has the homotopy type of the union. Below, the Čech
complex of discs around noisy samples of two touching rings has the
same Betti numbers as the union of the discs, counted independently from
a pixel image. The Vietoris-Rips complex at the same radius also fills
in triangles whose three discs meet pairwise but share no common point.
At :math:`r = 0.6` that closes the small ring's hole while the discs
still leave it open.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import betti_numbers, cech_complex, planar_betti_numbers, vietoris_rips_complex
from mathematicskit.topology.visualizers import plot_complex

rng = np.random.default_rng(1)
t, s = np.sort(rng.uniform(0, 2 * np.pi, 60)), np.sort(rng.uniform(0, 2 * np.pi, 40))
points = np.vstack([np.column_stack([np.cos(t), np.sin(t)]), np.column_stack([1.6 + 0.7 * np.cos(s), 0.7 * np.sin(s)])])
points += rng.normal(0, 0.03, points.shape)

xs = np.linspace(-1.6, 2.8, 700)
ys = np.linspace(-1.6, 1.6, 520)
gx, gy = np.meshgrid(xs, ys)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for col, r in enumerate((0.12, 0.3, 0.6)):
    union = np.zeros_like(gx, dtype=bool)
    for p in points:
        union |= np.hypot(gx - p[0], gy - p[1]) <= r
    C, R = cech_complex(points, r), vietoris_rips_complex(points, r)
    b_union, b_cech, b_rips = planar_betti_numbers(union), betti_numbers(C, field=2)[:2], betti_numbers(R, field=2)[:2]
    print(f"r = {r}: union of discs {b_union}, Čech {b_cech}, Rips {b_rips}")
    ax = axes[0, col]
    ax.contourf(gx, gy, union, levels=[0.5, 1.5], colors=["tab:blue"], alpha=0.3)
    plot_complex(C, ax=ax, alpha=0.4, vertex_size=4)
    ax.set_title(f"r = {r}: discs {b_union}, Čech {b_cech}")
    plot_complex(R, ax=axes[1, col], face_color="tab:orange", alpha=0.4, vertex_size=4)
    axes[1, col].set_title(f"Rips at the same r: {b_rips}")
for ax in axes.ravel():
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.axis("off")
fig.suptitle("Nerve theorem: the Čech complex has the homotopy type of the union of discs")
