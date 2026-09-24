r"""
Sylvester's problem: the smallest enclosing circle
========================================================

Finds the smallest circle containing a random point cloud with Welzl's
algorithm. The optimal circle always passes through two or three of the
points.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import min_enclosing_circle

# %%
# Random points and their enclosing circle
# -----------------------------------------------------

rng = np.random.default_rng(4)
points = rng.normal(size=(200, 2)) * [1.5, 0.8]
result = min_enclosing_circle(points)
on_circle = np.isclose(np.linalg.norm(points - result.center, axis=1), result.radius, atol=1e-9)
print(f"center {result.center.round(4)}, radius {result.radius:.4f}")
print(f"points on the circle: {int(on_circle.sum())}")

fig, ax = plt.subplots()
ax.plot(*points.T, ".", color="0.5")
ax.plot(*points[on_circle].T, "o", color="tab:red", label="points on the circle")
ax.add_patch(plt.Circle(result.center, result.radius, fill=False, color="tab:red"))
ax.set_aspect("equal")
ax.legend()
ax.set_title("Smallest enclosing circle (Welzl, 1991)")
