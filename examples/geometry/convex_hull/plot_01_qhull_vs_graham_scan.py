r"""
Qhull vs. Graham scan
===========================

Both methods find the same convex hull of a random point set --
scipy's Qhull-based implementation is the primary API, with a
hand-rolled Graham scan kept for comparison.
"""

# %%
import numpy as np

from mathematicskit.geometry import convex_hull, graham_scan
from mathematicskit.geometry.visualizers.plots import plot_convex_hull

# %%
# Generate random points and compute both hulls
# -----------------------------------------------------

rng = np.random.default_rng(0)
points = rng.uniform(-5, 5, size=(30, 2))

qhull_result = convex_hull(points)
graham_result = graham_scan(points)

print(f"Qhull:  {sorted(int(v) for v in qhull_result.vertices)}, area={qhull_result.volume:.4f}")
print(f"Graham: {sorted(int(v) for v in graham_result.vertices)}, area={graham_result.volume:.4f}")

# %%
# Visualize
# -----------------------------------------------------

plot_convex_hull(qhull_result)
