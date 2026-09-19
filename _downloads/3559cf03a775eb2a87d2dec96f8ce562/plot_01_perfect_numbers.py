r"""
Perfect numbers and Euler's totient function
===================================================

A perfect number equals the sum of its own proper divisors
(:math:`\sigma(n) = 2n`); this script finds the first few and plots
Euler's totient function over a range of ``n``.
"""

# %%
import numpy as np

from mathkit.number_theory import divisor_sum, euler_totient

# %%
# Find perfect numbers up to 10,000
# -----------------------------------------------------

perfect_numbers = [n for n in range(2, 10000) if divisor_sum(n) == 2 * n]
print("perfect numbers below 10000:", perfect_numbers)

# %%
# Euler's totient function
# -----------------------------------------------------

ns = np.arange(1, 31)
totients = [euler_totient(int(n)) for n in ns]
for n, phi in zip(ns, totients):
    print(f"phi({n}) = {phi}")
