r"""
Pólya's theorem: a drunk man finds his way home, a drunk bird may not
======================================================================

A simple random walk returns to its starting point with probability 1
in one and two dimensions but only about 34% of the time in three. The
simulated fraction of walks that have returned by step :math:`n` keeps
climbing toward 1 in 1D (slowly, like :math:`1 - 1/\sqrt{\pi n/2}`), crawls
upward in 2D, and levels off near 0.3405 in 3D.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.probability import random_walk_return_fraction, return_probability_1d, simple_random_walk

# %%
# A 2D walk
# -----------------------------------------------------

walk = simple_random_walk(n_walks=1, n_steps=5000, dim=2, seed=3)[0]
fig, ax = plt.subplots()
ax.plot(*walk.T, lw=0.6)
ax.plot(0, 0, "ro", label="origin")
ax.set_aspect("equal")
ax.legend()
ax.set_title("5000 steps of a 2D simple random walk")

# %%
# Return fractions by dimension
# -----------------------------------------------------

horizons = [10, 30, 100, 300, 1000]
fig, ax = plt.subplots()
for dim in (1, 2, 3):
    fracs = [random_walk_return_fraction(dim, n, n_walks=4000, seed=dim) for n in horizons]
    print(f"d={dim}: " + ", ".join(f"{f:.3f}" for f in fracs))
    ax.semilogx(horizons, fracs, "o-", label=f"d = {dim} (simulated)")
ax.semilogx(horizons, [return_probability_1d(n) for n in horizons], "k:", label="d = 1 exact")
ax.axhline(0.3405, color="0.5", ls="--", label="d = 3 limit 0.3405")
ax.set_xlabel("steps")
ax.set_ylabel("fraction returned to origin")
ax.legend()
ax.set_title("Pólya (1921): recurrence in d ≤ 2, transience in d = 3")
