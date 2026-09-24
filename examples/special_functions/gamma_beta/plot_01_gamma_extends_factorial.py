r"""
The gamma function extends the factorial
===============================================

Confirms Gamma(n) = (n-1)! at integers, Gamma(1/2) = sqrt(pi), and the
beta function's relationship to gamma.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import beta_function, gamma_function, log_gamma_function

# %%
# Gamma at integers and at 1/2
# -----------------------------------------------------

for n in range(1, 7):
    print(f"Gamma({n}) = {gamma_function(float(n)):.1f}, (n-1)! = {math.factorial(n - 1)}")

x = np.linspace(-3.95, 5.0, 2000)
g = gamma_function(x)
g[np.abs(g) > 30] = np.nan
fig, ax = plt.subplots()
ax.plot(x, g, label=r"$\Gamma(x)$")
ns = np.arange(1, 6)
ax.plot(ns, [math.factorial(n - 1) for n in ns], "ko", label=r"$(n-1)!$")
ax.axhline(0, color="gray", lw=0.5)
ax.set_ylim(-10, 25)
ax.set_xlabel("x")
ax.set_title("The gamma function interpolates the factorial")
ax.legend()

print(f"\nGamma(0.5) = {gamma_function(0.5):.6f}, sqrt(pi) = {np.sqrt(np.pi):.6f}")

# %%
# Log-gamma for large arguments where gamma itself would overflow
# -------------------------------------------------------------------

print(f"\nlog(Gamma(500)) = {log_gamma_function(500.0):.4f}")

# %%
# Beta function via the gamma-ratio identity
# -----------------------------------------------------

a, b = 3.0, 5.0
print(f"\nB({a},{b}) = {beta_function(a, b):.6f}")
print(f"Gamma({a})Gamma({b})/Gamma({a + b}) = {gamma_function(a) * gamma_function(b) / gamma_function(a + b):.6f}")
