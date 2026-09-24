r"""
Kahan's compensated summation
================================

Adding :math:`n` floating-point numbers one at a time can accumulate
error proportional to :math:`n`. William Kahan's 1965 algorithm carries
the rounding error of each addition forward in a correction term, so
the error stays at a few units in the last place however many terms
are added. This script sums :math:`0.1` repeatedly and compares the
naive loop, Kahan's algorithm, and NumPy's pairwise summation against
the exactly rounded result from :func:`math.fsum`.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import kahan_sum


def naive_sum(values):
    s = 0.0
    for v in values:
        s += v
    return s


# %%
# Relative error against the number of terms
# ----------------------------------------------

sizes = np.unique(np.logspace(1, 6, 16).astype(int))
eps = np.finfo(float).eps
errors = {"naive loop": [], "numpy.sum (pairwise)": [], "Kahan": []}
for n in sizes:
    values = [0.1] * int(n)
    exact = math.fsum(values)
    for name, total in (("naive loop", naive_sum(values)), ("numpy.sum (pairwise)", float(np.sum(values))), ("Kahan", kahan_sum(values))):
        errors[name].append(max(abs(total - exact) / exact, eps / 10))

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, marker in zip(errors, "os^"):
    ax.loglog(sizes, errors[name], marker + "-", label=name)
ax.loglog(sizes, sizes * eps, "--", color="gray", label=r"$n\,u$")
ax.set_xlabel("number of terms $n$")
ax.set_ylabel("relative error")
ax.set_title("Summing 0.1 repeatedly")
ax.legend(fontsize=8)
fig.tight_layout()

for name in errors:
    print(f"{name:22s} n = {sizes[-1]}: relative error {errors[name][-1]:.1e}")

plt.show()
