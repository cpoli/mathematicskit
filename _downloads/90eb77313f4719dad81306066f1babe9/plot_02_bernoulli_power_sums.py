r"""
Jacob Bernoulli's numbers and sums of powers
==================================================

Computes the Bernoulli numbers exactly, uses them to evaluate sums of
p-th powers in closed form, and repeats Bernoulli's own boast: the sum
of the tenth powers of the first 1000 integers.
"""

# %%
from mathematicskit.combinatorics import bernoulli_numbers, sum_of_powers

# %%
# The first Bernoulli numbers
# -----------------------------------------------------

for m, b in enumerate(bernoulli_numbers(12)):
    print(f"B_{m:<2d} = {b}")

# %%
# Closed-form power sums
# -----------------------------------------------------

for p in range(5):
    formula = sum_of_powers(100, p)
    direct = sum(k**p for k in range(1, 101))
    print(f"sum of k^{p} for k = 1..100: formula {formula}, direct {direct}")

print(f"\n1^10 + ... + 1000^10 = {sum_of_powers(1000, 10):,}")
