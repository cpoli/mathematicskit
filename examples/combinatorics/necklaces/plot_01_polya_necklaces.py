r"""
Pólya's enumeration theorem: necklaces and bracelets
==========================================================

Counts necklaces (up to rotation) and bracelets (up to rotation and
reflection) with Pólya's formula, and draws the 8 distinct two-colored
necklaces of 5 beads.
"""

# %%
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import count_bracelets, count_necklaces

# %%
# A table of counts
# -----------------------------------------------------

print("n   " + "".join(f"{f'k={k} neck/brac':>18}" for k in (2, 3, 4)))
for n in range(1, 11):
    print(f"{n:<4}" + "".join(f"{count_necklaces(n, k):>10}/{count_bracelets(n, k):<7}" for k in (2, 3, 4)))

# %%
# The 8 two-colored necklaces of 5 beads
# -----------------------------------------------------

n = 5
representatives = sorted({min(w[i:] + w[:i] for i in range(n)) for w in product((0, 1), repeat=n)})
angles = 2 * np.pi * np.arange(n) / n
fig, axes = plt.subplots(1, len(representatives), figsize=(12, 2))
for ax, word in zip(axes, representatives):
    ax.plot(np.cos(angles), np.sin(angles), color="0.7")
    ax.scatter(np.cos(angles), np.sin(angles), c=["k" if b else "w" for b in word], edgecolors="k", s=120, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle(f"count_necklaces(5, 2) = {count_necklaces(5, 2)}")
