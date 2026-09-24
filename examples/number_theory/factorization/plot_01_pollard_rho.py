r"""
Pollard's rho factorization
===================================================

Pollard's rho method finds a prime factor :math:`p` of :math:`n` in
roughly :math:`\sqrt p` steps, by waiting for a birthday-paradox
collision in the sequence :math:`x \mapsto x^2 + 1 \bmod p`. This script
factors the Fermat number :math:`F_6 = 2^{64}+1` and plots the step
count against the smallest prime factor for a batch of semiprimes.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import is_prime_miller_rabin, pollard_rho

# %%
# The sixth Fermat number
# -----------------------------------------------------

result = pollard_rho(2**64 + 1)
print(f"2^64 + 1 = {result.factor} * {result.cofactor}  ({result.iterations} iterations)")

# %%
# Steps grow like the square root of the smallest factor
# -----------------------------------------------------------

rng = np.random.default_rng(0)


def random_prime(lo, hi):
    while True:
        n = int(rng.integers(lo, hi)) | 1
        if is_prime_miller_rabin(n):
            return n


small_factors, steps = [], []
for bits in range(8, 34, 2):
    for _ in range(6):
        p = random_prime(2 ** (bits - 1), 2**bits)
        q = random_prime(2**40, 2**41)
        r = pollard_rho(p * q)
        small_factors.append(min(r.factor, r.cofactor))
        steps.append(r.iterations)
small_factors, steps = np.array(small_factors), np.array(steps)

fig, ax = plt.subplots()
ax.loglog(small_factors, steps, "o", alpha=0.6, label="Pollard rho")
ref = np.logspace(np.log10(small_factors.min()), np.log10(small_factors.max()), 50)
ax.loglog(ref, np.sqrt(np.pi * ref / 2), "--", color="firebrick", label="sqrt(pi p / 2)")
ax.set_xlabel("smallest prime factor p")
ax.set_ylabel("iterations")
ax.set_title("Rho finds p in about sqrt(p) steps")
ax.legend()
plt.show()
