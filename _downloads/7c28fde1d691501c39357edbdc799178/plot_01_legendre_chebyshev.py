r"""
Legendre and Chebyshev polynomials on [-1, 1]
===================================================

Plots the first Legendre polynomials P_n (weight 1) beside the first
Chebyshev polynomials T_n (weight 1/sqrt(1 - x^2)), checks each family's
orthogonality, and confirms Chebyshev's closed form T_n(cos t) = cos(nt),
whose equal ripples between -1 and 1 are its minimal-oscillation property.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import chebyshev_polynomial, inner_product, legendre_polynomial
from mathematicskit.special_functions.visualizers.plots import plot_polynomial_family

# %%
# The two families side by side
# -----------------------------------------------------

fig, (ax_p, ax_t) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
plot_polynomial_family(legendre_polynomial, degrees=[0, 1, 2, 3, 4], x_range=(-1.0, 1.0), ax=ax_p)
plot_polynomial_family(chebyshev_polynomial, degrees=[0, 1, 2, 3, 4], x_range=(-1.0, 1.0), ax=ax_t)
ax_p.set_title(r"Legendre $P_n(x)$, weight $1$")
ax_t.set_title(r"Chebyshev $T_n(x)$, weight $1/\sqrt{1-x^2}$")
fig.tight_layout()

# %%
# Orthogonality against each family's weight
# -----------------------------------------------------

legendre_ip = inner_product(lambda x: legendre_polynomial(2, x), lambda x: legendre_polynomial(3, x), lambda x: 1.0, -1.0, 1.0)
legendre_norm = inner_product(lambda x: legendre_polynomial(3, x), lambda x: legendre_polynomial(3, x), lambda x: 1.0, -1.0, 1.0)
cheb_weight = lambda x: 1.0 / np.sqrt(1.0 - x**2)  # noqa: E731
chebyshev_ip = inner_product(lambda x: chebyshev_polynomial(2, x), lambda x: chebyshev_polynomial(4, x), cheb_weight, -1.0, 1.0)
chebyshev_norm = inner_product(lambda x: chebyshev_polynomial(4, x), lambda x: chebyshev_polynomial(4, x), cheb_weight, -1.0, 1.0)

print(f"<P_2, P_3> = {legendre_ip:.2e}, <P_3, P_3> = {legendre_norm:.6f} (2/7 = {2 / 7:.6f})")
print(f"<T_2, T_4> = {chebyshev_ip:.2e}, <T_4, T_4> = {chebyshev_norm:.6f} (pi/2 = {np.pi / 2:.6f})")

# %%
# Chebyshev's closed form T_n(cos t) = cos(n t)
# -----------------------------------------------------

t = np.linspace(0.0, np.pi, 400)
for n in (3, 5, 8):
    err = np.max(np.abs(chebyshev_polynomial(n, np.cos(t)) - np.cos(n * t)))
    print(f"n = {n}: max |T_n(cos t) - cos(nt)| = {err:.1e}")
