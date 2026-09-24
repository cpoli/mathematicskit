r"""
Euler's polyhedron formula: V - E + F = 2
===============================================

Counts vertices, edges, and faces of the five Platonic solids and of
random convex polyhedra. Every one satisfies V - E + F = 2.
"""

# %%
import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from mathematicskit.geometry import polyhedron_counts

# %%
# The Platonic solids
# -----------------------------------------------------

phi = (1 + 5**0.5) / 2
cube = list(itertools.product((-1, 1), repeat=3))
solids = {
    "tetrahedron": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
    "cube": cube,
    "octahedron": [tuple(s if k == i else 0 for k in range(3)) for i in range(3) for s in (-1, 1)],
    "dodecahedron": cube + [p for a, b in itertools.product((-1, 1), repeat=2) for p in ((0, a / phi, b * phi), (a / phi, b * phi, 0), (b * phi, 0, a / phi))],
    "icosahedron": [p for a, b in itertools.product((-1, 1), repeat=2) for p in ((0, a, b * phi), (a, b * phi, 0), (b * phi, 0, a))],
}
for name, pts in solids.items():
    r = polyhedron_counts(np.array(pts, dtype=float))
    print(f"{name:12s}: V = {r.vertices:2d}, E = {r.edges:2d}, F = {r.faces:2d}, V - E + F = {r.euler_characteristic}")

# %%
# Random convex polyhedra
# -----------------------------------------------------

rng = np.random.default_rng(0)
for n in (10, 50, 250):
    r = polyhedron_counts(rng.normal(size=(n, 3)))
    print(f"hull of {n:3d} random points: V = {r.vertices:3d}, E = {r.edges:3d}, F = {r.faces:3d}, V - E + F = {r.euler_characteristic}")

pts = np.array(solids["dodecahedron"], dtype=float)
hull = ConvexHull(pts)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_trisurf(*pts.T, triangles=hull.simplices, alpha=0.5, edgecolor="k", linewidth=0.2)
ax.set_title("dodecahedron: 20 - 30 + 12 = 2")
