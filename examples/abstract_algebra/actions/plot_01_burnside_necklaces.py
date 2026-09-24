r"""
Burnside's lemma: counting necklaces
==========================================

Counts two-colored necklaces (up to rotation) and bracelets (up to
rotation and reflection) by averaging fixed points, and checks each
count against direct orbit enumeration.
"""

# %%
from itertools import product

import matplotlib.pyplot as plt

from mathematicskit.abstract_algebra import CyclicGroup, DihedralGroup, count_orbits, orbits

# %%
# Necklaces and bracelets of n beads
# -----------------------------------------------------

sizes = range(3, 10)
necklaces, bracelets = [], []
for n in sizes:
    words = [tuple(w) for w in product((0, 1), repeat=n)]
    rotate = lambda g, w: w[g:] + w[:g]  # noqa: E731
    permute = lambda g, w: tuple(w[g.index(i)] for i in range(len(w)))  # noqa: E731
    necklaces.append(count_orbits(CyclicGroup(n), words, rotate))
    bracelets.append(count_orbits(DihedralGroup(n), words, permute))
    assert necklaces[-1] == len(orbits(CyclicGroup(n), words, rotate))
    print(f"n = {n}: {2**n} colorings, {necklaces[-1]} necklaces, {bracelets[-1]} bracelets")

# %%
# Orbit counts grow like 2^n / n
# -----------------------------------------------------

fig, ax = plt.subplots()
ax.semilogy(list(sizes), [2**n for n in sizes], "o-", label="colorings $2^n$")
ax.semilogy(list(sizes), necklaces, "s-", label="necklaces (rotations)")
ax.semilogy(list(sizes), bracelets, "^-", label="bracelets (rotations + flips)")
ax.set_xlabel("number of beads n")
ax.set_ylabel("count")
ax.legend()
ax.set_title("Burnside's lemma")
