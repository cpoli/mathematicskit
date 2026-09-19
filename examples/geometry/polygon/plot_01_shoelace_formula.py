r"""
The shoelace formula: area and centroid of a regular hexagon
====================================================================

Confirms the shoelace formula's area against the closed-form regular-
polygon area formula.
"""

# %%
import numpy as np

from mathematicskit.geometry import polygon_area, polygon_centroid

# %%
# A regular hexagon of side length 1
# -----------------------------------------------------

n = 6
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
hexagon = np.column_stack([np.cos(angles), np.sin(angles)])

area = polygon_area(hexagon)
centroid = polygon_centroid(hexagon)
closed_form_area = 3.0 * np.sqrt(3.0) / 2.0

print(f"shoelace area: {area:.6f}, closed form (3*sqrt(3)/2): {closed_form_area:.6f}")
print(f"centroid: {centroid} (expected near the origin)")
