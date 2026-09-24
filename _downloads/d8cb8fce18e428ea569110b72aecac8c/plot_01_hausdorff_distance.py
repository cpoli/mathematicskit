r"""
The Hausdorff distance between shapes
===========================================

Compares a circle with approximating polygons. The Hausdorff distance
measures the worst-case gap between the two shapes and shrinks like
1/n^2 as the polygon gains sides; a single stray point makes it large
even though every other point matches.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import hausdorff_distance

# %%
# A circle against inscribed polygons
# -----------------------------------------------------

t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
circle = np.column_stack([np.cos(t), np.sin(t)])
sides = [4, 8, 16, 32, 64]
gaps = []
for n in sides:
    corners = np.column_stack([np.cos(2 * np.pi * np.arange(n + 1) / n), np.sin(2 * np.pi * np.arange(n + 1) / n)])
    polygon = np.vstack([np.linspace(corners[k], corners[k + 1], 200) for k in range(n)])
    gaps.append(hausdorff_distance(circle, polygon))
    print(f"{n:2d}-gon: Hausdorff distance {gaps[-1]:.5f}, exact 1 - cos(pi/n) = {1 - np.cos(np.pi / n):.5f}")

fig, ax = plt.subplots()
ax.loglog(sides, gaps, "o-", label="Hausdorff distance")
ax.loglog(sides, [np.pi**2 / (2 * n**2) for n in sides], "--", label=r"$\pi^2 / 2n^2$")
ax.set_xlabel("polygon sides n")
ax.legend()

# %%
# One outlier dominates
# -----------------------------------------------------

outlier = np.vstack([circle, [[3.0, 0.0]]])
print(f"\ncircle vs. circle plus one far point: {hausdorff_distance(circle, outlier):.3f}")
