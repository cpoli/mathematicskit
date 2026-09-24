r"""
Euler's officers: orthogonal Latin squares
================================================

Builds a pair of orthogonal Latin squares of order 5 and superimposes
them so that every (rank, regiment) pair appears exactly once, the
arrangement Euler's 36-officers problem asks for. It then checks by
brute force that no such pair exists for order 2.
"""

# %%
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import are_orthogonal, is_latin_square, orthogonal_latin_square_pair

# %%
# An orthogonal pair of order 5
# -----------------------------------------------------

ranks, regiments = orthogonal_latin_square_pair(5)
print("ranks:\n", ranks)
print("regiments:\n", regiments)
print(f"orthogonal: {are_orthogonal(ranks, regiments)}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, square, title in zip(axes, (ranks, regiments), ("rank", "regiment")):
    ax.imshow(square, cmap="tab10")
    for (i, j), v in np.ndenumerate(square):
        ax.text(j, i, str(v), ha="center", va="center", color="w")
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("Every (rank, regiment) pair appears exactly once")

# %%
# No orthogonal pair of order 2
# -----------------------------------------------------

squares = [np.array(rows) for rows in permutations(permutations(range(2)), 2)]
latin = [s for s in squares if is_latin_square(s)]
found = any(are_orthogonal(a, b) for a in latin for b in latin)
print(f"order 2: {len(latin)} Latin squares, orthogonal pair exists: {found}")
