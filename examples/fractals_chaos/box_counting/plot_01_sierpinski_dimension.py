r"""
Box-counting dimension of the Sierpinski triangle
=======================================================

Generates the Sierpinski triangle via the chaos game and estimates its
box-counting dimension, comparing against the known closed-form value
:math:`\log 3/\log 2 \approx 1.585`.
"""

# %%
import numpy as np

from mathkit.fractals_chaos import SierpinskiTriangle, box_counting_dimension
from mathkit.fractals_chaos.visualizers.plots import plot_box_counting, plot_ifs_points

# %%
# Generate the point cloud and estimate its dimension
# -----------------------------------------------------------

points = SierpinskiTriangle().generate(60000, seed=0)
result = box_counting_dimension(points)
print(f"estimated dimension: {result.dimension:.4f}")
print(f"exact value log(3)/log(2): {np.log(3.0) / np.log(2.0):.4f}")

# %%
# Visualize the point cloud and the log-log fit
# -----------------------------------------------------

plot_ifs_points(points)
plot_box_counting(result)
