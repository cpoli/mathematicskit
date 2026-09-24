r"""
Archimedes' method of exhaustion: squeezing pi
====================================================

Doubles the sides of inscribed and circumscribed regular polygons from
6 to 96, reproducing Archimedes' bounds 3 10/71 < pi < 3 1/7, and shows
that the gap shrinks by a factor of four with each doubling.
"""

# %%
import math

import matplotlib.pyplot as plt

from mathematicskit.calculus import archimedes_pi_bounds

# %%
# The 96-gon bounds
# -----------------------------------------------------

result = archimedes_pi_bounds(4)
for n, lo, hi in zip(result.sides, result.lower, result.upper):
    print(f"{n:3d} sides: {lo:.6f} < pi < {hi:.6f}")
print(f"Archimedes: {3 + 10 / 71:.6f} < pi < {3 + 1 / 7:.6f}")

# %%
# Convergence of the gap
# -----------------------------------------------------

result = archimedes_pi_bounds(15)
gaps = [hi - lo for lo, hi in zip(result.lower, result.upper)]
fig, ax = plt.subplots()
ax.loglog(result.sides, gaps, "o-", label="upper - lower")
ax.loglog(result.sides, [gaps[0] * (6 / n) ** 2 for n in result.sides], "--", label=r"$\propto n^{-2}$")
ax.axhline(abs(math.pi - 22 / 7), color="0.6", lw=0.8)
ax.set_xlabel("polygon sides n")
ax.set_ylabel("width of the bracket on pi")
ax.legend()
ax.set_title("Each doubling cuts the gap by 4")
