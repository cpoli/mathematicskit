r"""
The prime-counting function and the prime number theorem
================================================================

Sieves all primes up to 100,000 and compares the prime-counting
function :math:`\pi(n)` against the prime number theorem's
approximation :math:`n/\ln n`.
"""

# %%
import math

from mathkit.number_theory import is_prime_miller_rabin, sieve_of_eratosthenes
from mathkit.number_theory.visualizers.plots import plot_prime_counting

# %%
# Sieve and count
# -----------------------------------------------------

primes = sieve_of_eratosthenes(100000)
print(f"pi(100000) = {len(primes)} (prime number theorem estimate: {100000 / math.log(100000):.1f})")

# %%
# Cross-check the largest sieved prime with Miller-Rabin
# -----------------------------------------------------------

largest = int(primes[-1])
print(f"largest prime found: {largest}, Miller-Rabin agrees: {is_prime_miller_rabin(largest)}")

# %%
# Plot pi(n) vs. the prime number theorem approximation
# -----------------------------------------------------------

plot_prime_counting(2000)
