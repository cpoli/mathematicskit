r"""
Hermite and Laguerre polynomials on infinite domains
==========================================================

Plots the first Hermite polynomials H_n (weight e^{-x^2} on the whole
line) and Laguerre polynomials L_n (weight e^{-x} on the half-line),
checks their orthogonality over infinite intervals, and draws the
quantum harmonic oscillator states H_n(x) e^{-x^2/2} they describe.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import hermite_polynomial, inner_product, laguerre_polynomial
from mathematicskit.special_functions.visualizers.plots import plot_polynomial_family

# %%
# The two families
# -----------------------------------------------------

fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(10, 4))
plot_polynomial_family(hermite_polynomial, degrees=[0, 1, 2, 3], x_range=(-2.0, 2.0), ax=ax_h)
plot_polynomial_family(laguerre_polynomial, degrees=[0, 1, 2, 3], x_range=(0.0, 8.0), ax=ax_l)
ax_h.set_title(r"Hermite $H_n(x)$, weight $e^{-x^2}$ on $(-\infty, \infty)$")
ax_l.set_title(r"Laguerre $L_n(x)$, weight $e^{-x}$ on $[0, \infty)$")
fig.tight_layout()

# %%
# Orthogonality over infinite intervals
# -----------------------------------------------------

hermite_ip = inner_product(lambda x: hermite_polynomial(1, x), lambda x: hermite_polynomial(3, x), lambda x: np.exp(-(x**2)), -np.inf, np.inf)
hermite_norm = inner_product(lambda x: hermite_polynomial(3, x), lambda x: hermite_polynomial(3, x), lambda x: np.exp(-(x**2)), -np.inf, np.inf)
laguerre_ip = inner_product(lambda x: laguerre_polynomial(1, x), lambda x: laguerre_polynomial(3, x), lambda x: np.exp(-x), 0.0, np.inf)
laguerre_norm = inner_product(lambda x: laguerre_polynomial(3, x), lambda x: laguerre_polynomial(3, x), lambda x: np.exp(-x), 0.0, np.inf)

print(f"<H_1, H_3> = {hermite_ip:.2e}, <H_3, H_3> = {hermite_norm:.6f} (2^3 3! sqrt(pi) = {8 * 6 * math.sqrt(math.pi):.6f})")
print(f"<L_1, L_3> = {laguerre_ip:.2e}, <L_3, L_3> = {laguerre_norm:.6f} (expected 1)")

# %%
# Hermite functions: the quantum harmonic oscillator
# -----------------------------------------------------
# The normalized states psi_n(x) = H_n(x) e^{-x^2/2} / sqrt(2^n n! sqrt(pi)),
# drawn at their energy levels E_n = n + 1/2.

x = np.linspace(-5, 5, 500)
fig, ax = plt.subplots()
ax.plot(x, x**2 / 2, color="gray", lw=1)
for n in range(4):
    psi = hermite_polynomial(n, x) * np.exp(-(x**2) / 2) / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
    ax.plot(x, n + 0.5 + 0.8 * psi, label=rf"$\psi_{n}$, $E = {n}+\frac{{1}}{{2}}$")
ax.set_ylim(0, 5)
ax.set_xlabel("x")
ax.set_ylabel("energy / wavefunction (offset)")
ax.set_title("Harmonic oscillator states built from Hermite polynomials")
ax.legend(fontsize="small")
