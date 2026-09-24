r"""
Delaunay triangulation: empty circumcircles and fat triangles
===================================================================

Triangulates a random point set with
:func:`~mathematicskit.geometry.delaunay_triangulation`, checks
Delaunay's defining property (no point lies inside any triangle's
circumcircle), and shows on four points that choosing the Delaunay
diagonal maximizes the smallest angle and avoids needle-like
triangles.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import delaunay_triangulation, polygon_area
from mathematicskit.geometry.visualizers.plots import plot_triangulation


def circumcircle(a, b, c):
    """Center and radius of the circle through three points."""
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    sq = [p @ p for p in (a, b, c)]
    ux = (sq[0] * (b[1] - c[1]) + sq[1] * (c[1] - a[1]) + sq[2] * (a[1] - b[1])) / d
    uy = (sq[0] * (c[0] - b[0]) + sq[1] * (a[0] - c[0]) + sq[2] * (b[0] - a[0])) / d
    center = np.array([ux, uy])
    return center, np.linalg.norm(a - center)


def min_angle(points, simplices):
    """Smallest interior angle (degrees) over all triangles."""
    worst = 180.0
    for tri in simplices:
        p = points[tri]
        for k in range(3):
            u, v = p[(k + 1) % 3] - p[k], p[(k + 2) % 3] - p[k]
            cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
            worst = min(worst, np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return worst


# %%
# The empty-circle property
# -----------------------------------------------------

rng = np.random.default_rng(0)
points = rng.uniform(0, 10, size=(15, 2))
result = delaunay_triangulation(points)

violations = 0
for tri in result.simplices:
    center, radius = circumcircle(*points[tri])
    others = np.delete(points, tri, axis=0)
    violations += int(np.sum(np.linalg.norm(others - center, axis=1) < radius - 1e-9))
print(f"{result.simplices.shape[0]} triangles; points strictly inside a circumcircle: {violations}")

# %%
# Delaunay maximizes the smallest angle
# -----------------------------------------------------
# Four points can be triangulated in two ways, by either diagonal. The
# Delaunay choice avoids the thin pair of triangles and keeps the other
# diagonal's endpoints out of each circumcircle.

quad = np.array([[0.0, 0.0], [4.0, -0.6], [8.0, 0.0], [4.0, 1.2]])
delaunay_quad = delaunay_triangulation(quad).simplices
diagonal = set(delaunay_quad[0]) & set(delaunay_quad[1])
print(f"Delaunay uses diagonal {sorted(int(i) for i in diagonal)}")
# the other triangulation uses the opposite diagonal
other_quad = np.array([[0, 1, 2], [0, 2, 3]]) if diagonal == {1, 3} else np.array([[0, 1, 3], [1, 2, 3]])
print(f"smallest angle: Delaunay {min_angle(quad, delaunay_quad):.2f} deg, other diagonal {min_angle(quad, other_quad):.2f} deg")
print(f"both cover the same area: {polygon_area(quad):.4f}")

# %%
# The triangulation and its empty circumcircles
# -----------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
plot_triangulation(result, ax=axes[0])
for tri in result.simplices:
    center, radius = circumcircle(*points[tri])
    axes[0].add_patch(plt.Circle(center, radius, fill=False, color="tab:orange", lw=0.6, alpha=0.6))
axes[0].set_xlim(-2, 12)
axes[0].set_ylim(-2, 12)
axes[0].set_title("Every circumcircle is empty")
for ax, simplices, label in [(axes[1], delaunay_quad, "Delaunay diagonal"), (axes[2], other_quad, "other diagonal")]:
    ax.triplot(quad[:, 0], quad[:, 1], simplices, color="gray")
    ax.scatter(quad[:, 0], quad[:, 1], color="steelblue", zorder=2)
    for tri in simplices:
        center, radius = circumcircle(*quad[tri])
        ax.add_patch(plt.Circle(center, radius, fill=False, color="tab:orange", lw=0.8))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.set_title(f"{label}: min angle {min_angle(quad, simplices):.1f} deg")
plt.show()
