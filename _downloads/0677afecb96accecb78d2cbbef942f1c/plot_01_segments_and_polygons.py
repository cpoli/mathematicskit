r"""
Segment intersection and point-in-polygon
================================================

Finds where two segments cross, and tests a grid of points against a
concave ("L"-shaped) polygon.
"""

# %%
import numpy as np

from mathkit.geometry import point_in_polygon, segment_intersection

# %%
# Segment intersection
# -----------------------------------------------------

p = segment_intersection(np.array([0.0, 0.0]), np.array([4.0, 4.0]), np.array([0.0, 4.0]), np.array([4.0, 0.0]))
print(f"segments cross at {p}")

# %%
# Point-in-polygon on a concave "L" shape
# -----------------------------------------------------

l_shape = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
test_points = [(0.5, 0.5), (1.5, 1.5), (1.5, 0.5), (-0.5, 0.5)]
for tp in test_points:
    print(f"point {tp}: inside = {point_in_polygon(np.array(tp), l_shape)}")
