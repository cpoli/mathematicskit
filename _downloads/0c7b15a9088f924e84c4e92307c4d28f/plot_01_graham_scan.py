r"""
Graham's scan: the convex hull by an angular sweep
========================================================

Graham's scan sorts the points by angle around the lowest point, then
sweeps through them, dropping any point that would make a clockwise
turn. The picture shows the angular order the sweep follows and the
hull it produces; the result is checked against scipy's Qhull-based
:func:`~mathematicskit.geometry.convex_hull`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import convex_hull, graham_scan
from mathematicskit.geometry.visualizers.plots import plot_convex_hull

# %%
# Random points, the Graham scan, and the Qhull check
# -----------------------------------------------------

rng = np.random.default_rng(0)
points = rng.uniform(-5, 5, size=(30, 2))

graham_result = graham_scan(points)
qhull_result = convex_hull(points)

print(f"Graham: {sorted(int(v) for v in graham_result.vertices)}, area={graham_result.volume:.4f}")
print(f"Qhull:  {sorted(int(v) for v in qhull_result.vertices)}, area={qhull_result.volume:.4f}")

# %%
# The angular order of the sweep
# -----------------------------------------------------
# Rays from the pivot (the lowest point) to every other point, shaded by
# polar angle: the scan visits the points in this order and keeps only
# left turns.

pivot = points[int(np.lexsort((points[:, 0], points[:, 1]))[0])]
angles = np.arctan2(points[:, 1] - pivot[1], points[:, 0] - pivot[0])

fig, ax = plt.subplots(figsize=(6, 6))
cmap = plt.get_cmap("viridis")
for p, a in zip(points, angles):
    ax.plot([pivot[0], p[0]], [pivot[1], p[1]], color=cmap(a / np.pi), lw=0.6, alpha=0.7)
plot_convex_hull(graham_result, ax=ax)
ax.plot(*pivot, "*", color="k", ms=14, zorder=3, label="pivot")
ax.legend(loc="upper left")
ax.set_title("Graham scan: angular sweep around the lowest point")
plt.show()
