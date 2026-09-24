r"""
Euler-Maclaurin: correcting the trapezoidal rule
======================================================

Adds Euler-Maclaurin endpoint corrections, built from Bernoulli numbers
and odd derivatives at the endpoints, to the trapezoidal rule. Each
extra term raises the convergence order by two.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.calculus import euler_maclaurin_trapezoid

# %%
# Corrections for the integral of exp over [0, 1]
# -----------------------------------------------------

exact = np.e - 1
for m in range(4):
    value = euler_maclaurin_trapezoid(np.exp, 0.0, 1.0, 8, [np.exp] * m).value
    print(f"{m} correction terms, n = 8: error = {abs(value - exact):.2e}")

# %%
# Convergence with 0, 1, and 2 correction terms
# -----------------------------------------------------

ns = np.array([2, 4, 8, 16, 32, 64])
fig, ax = plt.subplots()
for m in range(3):
    errors = [abs(euler_maclaurin_trapezoid(np.exp, 0.0, 1.0, int(n), [np.exp] * m).value - exact) for n in ns]
    ax.loglog(ns, errors, "o-", label=f"{m} corrections: O(h^{2 * m + 2})")
ax.set_xlabel("n")
ax.set_ylabel("|error|")
ax.legend()
ax.set_title("Euler-Maclaurin corrections to the trapezoidal rule")
