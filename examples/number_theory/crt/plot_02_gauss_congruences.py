r"""
Gauss's Disquisitiones: congruences and the remainder theorem
==================================================================

Gauss's *Disquisitiones Arithmeticae* (1801) opens by defining
:math:`a \equiv b \pmod m` and computing with residues instead of
numbers. Its Articles 32-36 prove that congruences with pairwise
coprime moduli combine into one: working modulo :math:`m_1 m_2` is the
same as working modulo :math:`m_1` and :math:`m_2` at once. This script
checks Gauss's rules of congruence arithmetic, draws the residue grid
that pairs each class mod 35 with one class mod 5 and one mod 7, and
reduces a huge power with fast modular exponentiation.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import chinese_remainder_theorem, fast_mod_pow

# %%
# Congruences respect addition and multiplication
# -----------------------------------------------------
# If :math:`a \equiv a'` and :math:`b \equiv b' \pmod m`, then
# :math:`a + b \equiv a' + b'` and :math:`ab \equiv a'b'` (Articles 6-7).

m = 12
a, a2, b, b2 = 17, 5, 23, 11  # 17 = 5 and 23 = 11 (mod 12)
print(f"17 = 5 (mod 12): {(a - a2) % m == 0};  23 = 11 (mod 12): {(b - b2) % m == 0}")
print(f"sums agree mod 12: {(a + b) % m} = {(a2 + b2) % m};  products agree: {(a * b) % m} = {(a2 * b2) % m}")

# %%
# Combining congruences: residues mod 35 = pairs of residues mod 5 and 7
# -----------------------------------------------------------------------------

m1, m2 = 5, 7
grid = np.array([[chinese_remainder_theorem([r1, r2], [m1, m2]).residue for r2 in range(m2)] for r1 in range(m1)])
print("each residue mod 35 appears exactly once:", sorted(grid.ravel().tolist()) == list(range(m1 * m2)))

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.imshow(grid, cmap="viridis", origin="lower")
for r1 in range(m1):
    for r2 in range(m2):
        ax.text(r2, r1, str(grid[r1, r2]), ha="center", va="center", color="black" if grid[r1, r2] > 20 else "white", fontsize=11)
ax.set_xticks(range(m2))
ax.set_yticks(range(m1))
ax.set_xlabel("x mod 7")
ax.set_ylabel("x mod 5")
ax.set_title("Gauss's combination of congruences: x mod 35")
plt.show()

# %%
# Computing with residues: fast modular exponentiation
# -----------------------------------------------------
# :math:`2^{10^{18}}` has about :math:`3 \times 10^{17}` digits, yet its
# residue mod 35 follows from about 60 squarings, and agrees with the pair
# of residues mod 5 and mod 7 recombined.

e = 10**18
r35 = fast_mod_pow(2, e, 35)
r5, r7 = fast_mod_pow(2, e, 5), fast_mod_pow(2, e, 7)
print(f"2^(10^18) = {r35} (mod 35);  = {r5} (mod 5), {r7} (mod 7) -> recombined: {grid[r5, r7]}")
