r"""
Ramsey's theorem: R(3,3) = 6
==================================

Searches every red/blue coloring of the edges of K_5 and K_6. K_5 has
colorings with no single-colored triangle, such as the pentagon and
pentagram, but every one of the 32768 colorings of K_6 contains one.
"""

# %%
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import count_triangle_free_colorings, has_monochromatic_triangle

# %%
# Exhaustive search
# -----------------------------------------------------

for n in range(3, 7):
    total = 2 ** (n * (n - 1) // 2)
    print(f"K_{n}: {count_triangle_free_colorings(n)} of {total} colorings avoid a monochromatic triangle")

# %%
# The extremal coloring of K_5
# -----------------------------------------------------

coloring = {(i, j): int((j - i) % 5 in (1, 4)) for i, j in combinations(range(5), 2)}
print(f"pentagon/pentagram coloring has a monochromatic triangle: {has_monochromatic_triangle(5, coloring)}")
points = np.array([(np.sin(2 * np.pi * k / 5), np.cos(2 * np.pi * k / 5)) for k in range(5)])
fig, ax = plt.subplots()
for (i, j), color in coloring.items():
    ax.plot(*points[[i, j]].T, color=("tab:red" if color else "tab:blue"), lw=2)
ax.plot(*points.T, "ko")
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("K_5 with no one-colored triangle")
