r"""
The Jordan curve theorem and ray casting: inside or outside?
==================================================================

A simple closed polygon splits the plane into an inside and an outside
(the Jordan curve theorem). A ray cast from a point crosses the
boundary an odd number of times exactly when the point is inside. The
crossings are found edge by edge with
:func:`~mathematicskit.geometry.segment_intersection`, and their parity
agrees with :func:`~mathematicskit.geometry.point_in_polygon` on a
concave, spiral-like polygon.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import point_in_polygon, segment_intersection

# %%
# A concave polygon and a few test points
# -----------------------------------------------------

polygon = np.array(
    [[0, 0], [8, 0], [8, 7], [2, 7], [2, 3], [5, 3], [5, 5], [4, 5], [4, 4], [3, 4], [3, 6], [7, 6], [7, 1], [1, 1], [1, 8], [0, 8]],
    dtype=float,
)
test_points = np.array([[0.5, 2.6], [2.5, 4.6], [3.5, 3.5], [4.5, 4.2], [6.0, 5.4], [7.5, 1.5], [-1.0, 6.5]])
ray_end_x = polygon[:, 0].max() + 1.0

# %%
# Count crossings of a ray in the +x direction
# -----------------------------------------------------

crossings = []
for p in test_points:
    far = np.array([ray_end_x, p[1]])
    hits = [segment_intersection(p, far, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
    hits = [h for h in hits if h is not None]
    crossings.append(hits)
    inside = point_in_polygon(p, polygon)
    print(f"point ({p[0]:4.1f}, {p[1]:.1f}): {len(hits)} crossings -> {'odd' if len(hits) % 2 else 'even'}, point_in_polygon = {inside}")

# %%
# Rays, crossings, and the inside/outside regions
# -----------------------------------------------------

xs = np.linspace(-1.5, 9, 220)
ys = np.linspace(-1, 9, 220)
X, Y = np.meshgrid(xs, ys)
inside_grid = np.array([point_in_polygon(np.array([x, y]), polygon) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig, ax = plt.subplots(figsize=(7, 7))
ax.contourf(X, Y, inside_grid, levels=[-0.5, 0.5, 1.5], colors=["white", "lightsteelblue"])
closed = np.vstack([polygon, polygon[:1]])
ax.plot(*closed.T, "k", lw=1.5)
for p, hits in zip(test_points, crossings):
    color = "tab:green" if len(hits) % 2 else "tab:red"
    ax.plot([p[0], ray_end_x], [p[1], p[1]], color=color, lw=1, ls="--")
    ax.plot(*p, "o", color=color, ms=8)
    if hits:
        ax.plot(*np.array(hits).T, "x", color=color, ms=7)
    ax.text(ray_end_x + 0.1, p[1], str(len(hits)), color=color, va="center")
ax.set_aspect("equal")
ax.set_title("Ray casting: odd crossings (green) = inside")
plt.show()
