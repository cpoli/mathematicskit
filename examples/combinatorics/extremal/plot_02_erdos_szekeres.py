r"""
The Erdős-Szekeres theorem: monotone subsequences
=======================================================

Any sequence of (r-1)(s-1)+1 distinct numbers contains an increasing
subsequence of length r or a decreasing one of length s. This example
finds the longest monotone subsequences of random permutations and shows
that their length grows like 2 sqrt(n).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import longest_decreasing_subsequence, longest_increasing_subsequence

# %%
# Every 10-term sequence has a monotone run of length 4
# -----------------------------------------------------

rng = np.random.default_rng(1)
for _ in range(5):
    seq = rng.permutation(10).tolist()
    up, down = longest_increasing_subsequence(seq), longest_decreasing_subsequence(seq)
    print(f"{seq}: increasing {up}, decreasing {down}")

# %%
# A sequence that meets the bound exactly
# -----------------------------------------------------

sharp = [7, 8, 9, 4, 5, 6, 1, 2, 3]
print(f"\n{sharp}: longest runs {len(longest_increasing_subsequence(sharp))} and {len(longest_decreasing_subsequence(sharp))}")

# %%
# Growth for random permutations
# -----------------------------------------------------

sizes = [10, 30, 100, 300, 1000, 3000]
means = [np.mean([len(longest_increasing_subsequence(rng.permutation(n).tolist())) for _ in range(40)]) for n in sizes]
fig, ax = plt.subplots()
ax.loglog(sizes, means, "o-", label="mean longest increasing subsequence")
ax.loglog(sizes, 2 * np.sqrt(sizes), "--", label=r"$2\sqrt{n}$")
ax.loglog(sizes, np.sqrt(sizes), ":", label=r"Erdős-Szekeres guarantee $\sqrt{n}$")
ax.set_xlabel("n")
ax.legend()
