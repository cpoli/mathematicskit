r"""
The Riemann zeta function and Euler's product
==================================================

Plots zeta(s) on the real line, including its analytic continuation to
s < 1 with trivial zeros at the negative even integers, and shows
Euler's product over primes converging to zeta(2) = pi^2/6.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import euler_product, riemann_zeta

# %%
# zeta on the real line
# -----------------------------------------------------

s = np.concatenate([np.linspace(-9, 0.95, 500), np.linspace(1.05, 6, 200)])
values = riemann_zeta(s)
fig, ax = plt.subplots()
ax.plot(s[s < 1], values[s < 1], "C0")
ax.plot(s[s > 1], values[s > 1], "C0")
ax.plot([-2, -4, -6, -8], [0, 0, 0, 0], "ro", label="trivial zeros")
ax.axhline(0, color="gray", lw=0.5)
ax.set_ylim(-2, 4)
ax.set_xlabel("s")
ax.set_ylabel(r"$\zeta(s)$")
ax.legend()
print(f"zeta(2) = {riemann_zeta(2.0):.12f}, pi^2/6 = {math.pi**2 / 6:.12f}")
print(f"zeta(0) = {riemann_zeta(0.0):.6f}, zeta(-1) = {riemann_zeta(-1.0):.6f} (= -1/12)")

# %%
# Euler's product over primes
# -----------------------------------------------------

for bound in (10, 100, 1000, 10000):
    product = euler_product(2.0, bound)
    print(f"primes <= {bound:5d}: product = {product:.8f}, error = {abs(product - math.pi**2 / 6):.1e}")
