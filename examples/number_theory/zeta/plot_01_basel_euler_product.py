r"""
The Basel problem and Euler's product formula
===================================================

Euler showed in 1735 that :math:`\sum 1/n^2 = \pi^2/6`, and in 1737
that the same sum is a product over the primes,
:math:`\zeta(s) = \prod_p (1 - p^{-s})^{-1}`. This script compares how
fast the partial sums and the truncated Euler product approach
:math:`\pi^2/6`, and shows the product diverging at :math:`s = 1`.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import zeta

from mathematicskit.number_theory import euler_product

# %%
# Partial sums vs. the Euler product at s = 2
# -----------------------------------------------------

limits = np.unique(np.logspace(1, 6, 30).astype(int))
partial_sums = np.array([np.sum(1.0 / np.arange(1, n + 1) ** 2) for n in limits])
products = np.array([euler_product(2.0, int(n)) for n in limits])
print(f"pi^2/6               = {math.pi**2 / 6:.10f}")
print(f"scipy zeta(2)        = {zeta(2.0):.10f}")
print(f"Euler product p<=1e6 = {products[-1]:.10f}")

fig, ax = plt.subplots()
ax.loglog(limits, math.pi**2 / 6 - partial_sums, "o-", label="sum over n <= N")
ax.loglog(limits, np.abs(math.pi**2 / 6 - products), "s-", label="product over primes p <= N")
ax.set_xlabel("N")
ax.set_ylabel("|error| vs. pi^2 / 6")
ax.set_title("Two roads to zeta(2)")
ax.legend()

# %%
# At s = 1 the product diverges: infinitely many primes
# -----------------------------------------------------------
# Mertens' third theorem gives :math:`\prod_{p \le N}(1-1/p)^{-1} \sim
# e^{\gamma}\ln N`; a finite set of primes would give a finite product.

harmonic = np.array([euler_product(1.0, int(n)) for n in limits])
fig, ax = plt.subplots()
ax.semilogx(limits, harmonic, "o-", label="product over p <= N at s = 1")
ax.semilogx(limits, np.exp(np.euler_gamma) * np.log(limits), "--", color="firebrick", label="e^gamma ln N")
ax.set_xlabel("N")
ax.legend()
ax.set_title("Euler's proof that the primes are infinite")
plt.show()
