r"""
Perfect numbers and the divisor-sum function
===================================================

A perfect number equals the sum of its own proper divisors
(:math:`\sigma(n) = 2n`). This script finds the perfect numbers below
10,000 with
:func:`~mathematicskit.number_theory.systems.totient.divisor_sum` and
checks Euclid's form :math:`2^{p-1}(2^p - 1)` for each.
"""

# %%
from mathematicskit.number_theory import divisor_sum

# %%
# Find perfect numbers up to 10,000
# -----------------------------------------------------

perfect_numbers = [n for n in range(2, 10000) if divisor_sum(n) == 2 * n]
print("perfect numbers below 10000:", perfect_numbers)

# %%
# Each has Euclid's form 2^(p-1) (2^p - 1)
# -----------------------------------------------------

for n in perfect_numbers:
    p = (n & -n).bit_length()  # 2^(p-1) is the largest power of 2 dividing n
    print(f"  {n} = 2^{p - 1} * (2^{p} - 1) -> {2 ** (p - 1) * (2**p - 1) == n}")
