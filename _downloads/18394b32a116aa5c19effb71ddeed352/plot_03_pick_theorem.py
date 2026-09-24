r"""
Pick's theorem: area from lattice points
==============================================

Counts the lattice points inside and on the boundary of polygons with
integer vertices, and checks Pick's formula A = I + B/2 - 1 against the
shoelace area.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import lattice_point_counts, point_in_polygon

# %%
# Several lattice polygons
# -----------------------------------------------------

polygons = {
    "rectangle": [[0, 0], [4, 0], [4, 3], [0, 3]],
    "triangle": [[0, 0], [7, 2], [3, 6]],
    "non-convex": [[0, 0], [6, 0], [6, 5], [3, 2], [0, 5]],
}
for name, vertices in polygons.items():
    r = lattice_point_counts(vertices)
    print(f"{name:10s}: I = {r.interior:2d}, B = {r.boundary:2d}, I + B/2 - 1 = {r.pick_area:5.1f}, shoelace area = {r.area:5.1f}")

# %%
# Picture of the non-convex case
# -----------------------------------------------------

v = np.array(polygons["non-convex"], dtype=float)
fig, ax = plt.subplots()
ax.fill(*v.T, alpha=0.3)
for x in range(0, 7):
    for y in range(0, 6):
        inside = point_in_polygon(np.array([x, y], dtype=float), v)
        ax.plot(x, y, "o", color="tab:green" if inside else "0.8", ms=5)
ax.set_aspect("equal")
ax.set_title("Pick: area = interior points + boundary points / 2 - 1")
