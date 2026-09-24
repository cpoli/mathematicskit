r"""
Stirling's numbers: cycles, set partitions, and Bell numbers
=================================================================

Checks by brute force that Stirling numbers of the first kind count
permutations by their number of cycles and that Stirling numbers of the
second kind count partitions of a labeled set into k blocks. It then
recovers Stirling's own use of them, converting ordinary powers into
falling factorials, and sums the second kind to get the Bell numbers.
"""

# %%
from itertools import permutations, product
from math import factorial, perm

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import bell_number, stirling_first_kind, stirling_second_kind


def _cycle_count(p):
    seen, cycles = set(), 0
    for start in range(len(p)):
        if start not in seen:
            cycles += 1
            i = start
            while i not in seen:
                seen.add(i)
                i = p[i]
    return cycles


def _block_count(labels):
    # A restricted-growth string (label i is at most 1 + max of earlier labels)
    # encodes each set partition exactly once.
    if labels[0] != 0 or any(b > max(labels[:i]) + 1 for i, b in enumerate(labels) if i):
        return None
    return max(labels) + 1


# %%
# First kind: permutations of 5 elements by number of cycles
# ---------------------------------------------------------------

n = 5
by_cycles = [0] * (n + 1)
for p in permutations(range(n)):
    by_cycles[_cycle_count(p)] += 1
print("enumerated :", by_cycles)
print("[5 k]      :", [stirling_first_kind(n, k) for k in range(n + 1)])
print(f"sum = {sum(by_cycles)} = 5!")

# %%
# Second kind: partitions of {1,...,5} into k blocks
# ---------------------------------------------------------------

by_blocks = [0] * (n + 1)
for labels in product(range(n), repeat=n):
    k = _block_count(labels)
    if k is not None:
        by_blocks[k] += 1
print("enumerated :", by_blocks)
print("{5 k}      :", [stirling_second_kind(n, k) for k in range(n + 1)])
print(f"Bell number B_5 = {bell_number(n)} = {sum(by_blocks)} set partitions")

# %%
# Stirling's conversion between powers and falling factorials
# ---------------------------------------------------------------

x = 7
for m in range(1, 6):
    via_second = sum(stirling_second_kind(m, k) * perm(x, k) for k in range(m + 1))
    via_first = sum(stirling_first_kind(m, k, signed=True) * x**k for k in range(m + 1))
    print(f"m={m}: x^m = {x**m} = sum S2*x_(k) = {via_second};  x_(m) = {perm(x, m)} = sum s1*x^k = {via_first}")
    assert via_second == x**m and via_first == perm(x, m)

# %%
# The two triangles side by side
# ---------------------------------------------------------------

N = 8
first = np.array([[stirling_first_kind(i, k) for k in range(N + 1)] for i in range(N + 1)], dtype=float)
second = np.array([[stirling_second_kind(i, k) for k in range(N + 1)] for i in range(N + 1)], dtype=float)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, table, name in zip(axes, [first, second], ["first kind (cycles)", "second kind (blocks)"]):
    shown = np.where(table > 0, np.log10(np.maximum(table, 1)), np.nan)
    ax.imshow(shown, cmap="viridis")
    for i in range(N + 1):
        for k in range(i + 1):
            if table[i, k] > 0:
                dark = shown[i, k] > 0.6 * np.nanmax(shown)
                ax.text(k, i, f"{int(table[i, k])}", ha="center", va="center", fontsize=6, color="black" if dark else "white")
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_title(f"Stirling numbers of the {name}")
fig.tight_layout()
print(f"\nrow sums: first kind -> n! = {factorial(N)}, second kind -> Bell B_{N} = {bell_number(N)}")
