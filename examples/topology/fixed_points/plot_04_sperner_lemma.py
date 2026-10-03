r"""
Sperner's lemma (1928)
======================

Subdivide a triangle and label each vertex 0, 1 or 2, with the corners
labelled 0, 1, 2 and each side vertex labelled like one of its side's
endpoints (interior labels are free). Emanuel Sperner proved that an
*odd* number of small triangles carry all three labels. Picture labels
0-1 edges as doors: one enters from the bottom side, and following
doors from room to room can only stop in a fully labelled room.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import fully_labeled_triangles, random_sperner_labeling, triangle_grid

grid = triangle_grid(12)
counts = [len(fully_labeled_triangles(grid, random_sperner_labeling(grid, seed=s))) for s in range(2000)]
print("fully labelled triangles over 2000 random labellings: all odd?", all(c % 2 == 1 for c in counts))

# %%
# One labelling, its doors, and its fully labelled triangles
# ----------------------------------------------------------

labels = random_sperner_labeling(grid, seed=7)
full = fully_labeled_triangles(grid, labels)
xy = grid.cartesian
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), width_ratios=(1.2, 1))
ax = axes[0]
ax.triplot(*xy.T, grid.triangles, color="0.75", lw=0.6)
for tri in grid.triangles[full]:
    ax.add_patch(plt.Polygon(xy[tri], color="gold", ec="k", zorder=2))
for tri in grid.triangles:
    for a, b in ((0, 1), (1, 2), (0, 2)):
        if {labels[tri[a]], labels[tri[b]]} == {0, 1}:
            ax.plot(*xy[[tri[a], tri[b]]].T, color="tab:purple", lw=2.5, zorder=1)
for k, color in enumerate(("tab:red", "tab:green", "tab:blue")):
    ax.scatter(*xy[labels == k].T, s=25, color=color, zorder=3, label=f"label {k}")
ax.plot([], [], color="tab:purple", lw=2.5, label="0-1 door")
ax.set_aspect("equal")
ax.axis("off")
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"{len(full)} fully labelled triangles (gold)")

ax = axes[1]
values, freq = np.unique(counts, return_counts=True)
ax.bar(values, freq, color="tab:orange")
ax.set_xlabel("number of fully labelled triangles")
ax.set_ylabel("labellings")
ax.set_title("always odd")
fig.suptitle("Sperner (1928)")
