r"""
Fibonacci's rabbits and domino tilings
============================================

Reproduces the rabbit problem from Fibonacci's 1202 *Liber Abaci*,
checks that the same numbers count domino tilings of a 2-by-n strip,
and shows the ratio of successive terms approaching the golden ratio.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import domino_tilings, fibonacci

# %%
# Rabbit pairs month by month
# -----------------------------------------------------

# Liber Abaci starts from one pair and counts the pairs after each month.
print("month:       " + " ".join(f"{m:4d}" for m in range(0, 13)))
print("rabbit pairs:" + " ".join(f"{fibonacci(m + 2):4d}" for m in range(0, 13)))
print(f"after a year: {fibonacci(14)} pairs, Fibonacci's own answer")

# %%
# Tilings of a 2-by-n strip
# -----------------------------------------------------

for n in range(1, 8):
    print(f"2 x {n} strip: {domino_tilings(n)} tilings")

# %%
# Ratios converge to the golden ratio
# -----------------------------------------------------

n = np.arange(2, 30)
ratios = [fibonacci(k + 1) / fibonacci(k) for k in n]
phi = (1 + np.sqrt(5)) / 2
fig, ax = plt.subplots()
ax.semilogy(n, [abs(r - phi) for r in ratios], "o-")
ax.set_xlabel("n")
ax.set_ylabel(r"$|F_{n+1}/F_n - \varphi|$")
ax.set_title("Successive ratios approach the golden ratio")
