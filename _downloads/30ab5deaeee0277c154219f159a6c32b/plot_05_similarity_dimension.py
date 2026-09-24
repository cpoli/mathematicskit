r"""
Hausdorff and Moran: the similarity dimension
===================================================

Solves Moran's equation sum r_i^D = 1 for several self-similar sets and
compares the answers with box-counting estimates computed from the
sets themselves.
"""

# %%
import numpy as np

from mathematicskit.fractals_chaos import (
    SierpinskiCarpet,
    SierpinskiTriangle,
    box_counting_dimension,
    koch_curve,
    similarity_dimension,
)

# %%
# Moran's equation against box counting
# -----------------------------------------------------

cantor_points = np.array([[sum(int(d) * 2 * 3.0 ** -(k + 1) for k, d in enumerate(f"{i:012b}")), 0.0] for i in range(4096)])
cases = {
    "Cantor set": ([1 / 3] * 2, cantor_points),
    "Koch curve": ([1 / 3] * 4, koch_curve(7)),
    "Sierpinski triangle": ([1 / 2] * 3, SierpinskiTriangle().generate(200000)),
    "Sierpinski carpet": ([1 / 3] * 8, SierpinskiCarpet().generate(300000)),
}
for name, (ratios, points) in cases.items():
    if name == "Cantor set":
        points = np.column_stack([points[:, 0], np.zeros(len(points))])
        sizes = np.logspace(-1, -3.3, 10)
        estimate = np.polyfit(np.log(1 / sizes), np.log([len(np.unique(np.floor(points[:, 0] / s))) for s in sizes]), 1)[0]
    else:
        estimate = box_counting_dimension(points).dimension
    print(f"{name:20s} similarity dimension {similarity_dimension(ratios):.4f}, box counting {estimate:.4f}")

# %%
# Unequal ratios
# -----------------------------------------------------

print(f"\nratios (1/2, 1/4, 1/4): D = {similarity_dimension([0.5, 0.25, 0.25]):.4f}")
print(f"ratios (0.6, 0.3):      D = {similarity_dimension([0.6, 0.3]):.4f}")
