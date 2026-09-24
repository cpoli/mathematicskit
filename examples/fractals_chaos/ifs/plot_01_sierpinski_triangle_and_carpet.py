r"""
Sierpiński's triangle and carpet
======================================

Sierpiński's triangle (1915) removes the middle quarter of a filled
triangle and repeats on the three corner triangles; his carpet (1916)
removes the middle ninth of a square and repeats on the eight remaining
squares. Both limits have zero area and a dimension strictly between 1
and 2. Here both are drawn with the chaos game.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import SierpinskiCarpet, SierpinskiTriangle
from mathematicskit.fractals_chaos.visualizers.plots import plot_ifs_points

# %%
# The two sets
# ------------

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_ifs_points(SierpinskiTriangle().generate(60000, seed=0), ax=axes[0], color="steelblue")
axes[0].set_title("Sierpiński triangle")
plot_ifs_points(SierpinskiCarpet().generate(80000, seed=0), ax=axes[1], color="firebrick")
axes[1].set_title("Sierpiński carpet")
fig.tight_layout()

# %%
# Area vanishes, dimension is fractional
# --------------------------------------
#
# After :math:`n` removals the triangle keeps :math:`(3/4)^n` of its
# area and the carpet :math:`(8/9)^n`; both tend to zero. Three copies at
# scale 1/2 and eight copies at scale 1/3 give dimensions
# :math:`\log 3/\log 2` and :math:`\log 8/\log 3`.

for n in (1, 5, 20, 50):
    print(f"n = {n:2d}: triangle area {(3 / 4) ** n:.2e}, carpet area {(8 / 9) ** n:.2e}")
print(f"dimension of triangle: {np.log(3) / np.log(2):.4f}, carpet: {np.log(8) / np.log(3):.4f}")
