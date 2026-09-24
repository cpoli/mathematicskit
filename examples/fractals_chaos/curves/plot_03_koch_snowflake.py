r"""
The Koch snowflake: infinite perimeter, finite area
=========================================================

Builds the Koch snowflake. Each refinement multiplies the perimeter by
4/3, so it grows without bound, while the enclosed area converges to
8/5 of the starting triangle.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import box_counting_dimension, koch_curve, koch_snowflake, similarity_dimension

# %%
# The first refinements
# -----------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
triangle_area = np.sqrt(3) / 4
for ax, order in zip(axes, range(4)):
    pts = koch_snowflake(order)
    perimeter = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    area = 0.5 * abs(np.dot(pts[:-1, 0], pts[1:, 1]) - np.dot(pts[1:, 0], pts[:-1, 1]))
    ax.fill(*pts.T, alpha=0.4)
    ax.plot(*pts.T, lw=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"perimeter {perimeter:.2f}\narea / triangle {area / triangle_area:.3f}")
print(f"limiting area ratio 8/5 = {8 / 5}")

# %%
# Dimension of the Koch curve
# -----------------------------------------------------

print(f"similarity dimension log 4 / log 3 = {similarity_dimension([1 / 3] * 4):.4f}")
print(f"box-counting estimate on order 7: {box_counting_dimension(koch_curve(7)).dimension:.4f}")
