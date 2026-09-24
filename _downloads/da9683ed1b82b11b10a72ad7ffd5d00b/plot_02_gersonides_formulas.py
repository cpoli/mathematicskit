r"""
Gersonides' counting formulas
===================================

Levi ben Gershon's 1321 *Maaseh Hoshev* proved by induction that n
objects can be arranged in n! orders, that k of them can be arranged in
n!/(n-k)! ways, and that k can be chosen in n!/(k!(n-k)!) ways. This
example checks each formula against brute-force enumeration.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.combinatorics import combinations_count, generate_combinations, generate_permutations, permutations_count

# %%
# Formula against enumeration
# -----------------------------------------------------

items = "ABCDE"
n = len(items)
print(f"arrangements of {n} objects: formula {permutations_count(n)}, listed {len(generate_permutations(items))}")
for k in range(n + 1):
    ordered = permutations_count(n, k)
    unordered = combinations_count(n, k)
    listed = len(generate_combinations(items, k))
    print(f"k = {k}: P(n,k) = {ordered:3d}, C(n,k) = {unordered:2d} (listed {listed}), P/C = k! = {ordered // unordered}")

# %%
# Arrangements grow much faster than selections
# -----------------------------------------------------

ks = range(0, 11)
fig, ax = plt.subplots()
ax.semilogy(list(ks), [permutations_count(10, k) for k in ks], "o-", label="ordered: P(10, k)")
ax.semilogy(list(ks), [combinations_count(10, k) for k in ks], "s-", label="unordered: C(10, k)")
ax.set_xlabel("k")
ax.set_ylabel("count")
ax.legend()
ax.set_title("Choosing k of 10 objects")
