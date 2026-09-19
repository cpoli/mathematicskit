r"""
Delaunay triangulation and its Voronoi dual
==================================================

Builds both a Delaunay triangulation and its dual Voronoi diagram for
the same random point set.
"""

# %%
import numpy as np

from mathematicskit.geometry import delaunay_triangulation
from mathematicskit.geometry.visualizers.plots import plot_triangulation, plot_voronoi

# %%
# Random points
# -----------------------------------------------------

rng = np.random.default_rng(0)
points = rng.uniform(0, 10, size=(15, 2))

# %%
# Delaunay triangulation
# -----------------------------------------------------

result = delaunay_triangulation(points)
print(f"{result.simplices.shape[0]} triangles from {points.shape[0]} points")

plot_triangulation(result)

# %%
# The dual Voronoi diagram
# -----------------------------------------------------

plot_voronoi(points)
