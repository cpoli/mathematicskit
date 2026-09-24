r"""
Mersenne primes and the Lucas-Lehmer test
===================================================

The Lucas-Lehmer test decides whether :math:`M_p = 2^p - 1` is prime
with only :math:`p - 2` modular squarings. This script rediscovers
every Mersenne prime with exponent below 1300 -- including
:math:`M_{521}` through :math:`M_{1279}`, found by Raphael Robinson on
the SWAC computer in 1952 -- and plots the exponents, which grow roughly
geometrically.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import lucas_lehmer, sieve_of_eratosthenes

# %%
# Search every prime exponent below 1300
# -----------------------------------------------------

exponents = [int(p) for p in sieve_of_eratosthenes(1300) if lucas_lehmer(int(p))]
print("Mersenne prime exponents:", exponents)
print(f"M_1279 has {len(str(2**1279 - 1))} decimal digits")
print("M_11 = 2047 = 23 * 89 is composite:", not lucas_lehmer(11))

# %%
# Exponents grow geometrically
# -----------------------------------------------------

fig, ax = plt.subplots()
ax.semilogy(np.arange(1, len(exponents) + 1), exponents, "o-")
ax.set_xlabel("n (n-th Mersenne prime)")
ax.set_ylabel("exponent p")
ax.set_title("Mersenne prime exponents below 1300")
plt.show()
