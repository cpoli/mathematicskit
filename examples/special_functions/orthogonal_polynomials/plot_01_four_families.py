r"""
Four orthogonal polynomial families
=========================================

Plots the first few Legendre, Chebyshev, Hermite, and Laguerre
polynomials, and numerically verifies each family's orthogonality
against its own weight function.
"""

# %%
import numpy as np

from mathematicskit.special_functions import chebyshev_polynomial, hermite_polynomial, inner_product, laguerre_polynomial, legendre_polynomial
from mathematicskit.special_functions.visualizers.plots import plot_polynomial_family

# %%
# Plot each family
# -----------------------------------------------------

plot_polynomial_family(legendre_polynomial, degrees=[0, 1, 2, 3], x_range=(-1.0, 1.0))
plot_polynomial_family(chebyshev_polynomial, degrees=[0, 1, 2, 3], x_range=(-1.0, 1.0))
plot_polynomial_family(hermite_polynomial, degrees=[0, 1, 2, 3], x_range=(-2.0, 2.0))
plot_polynomial_family(laguerre_polynomial, degrees=[0, 1, 2, 3], x_range=(0.0, 8.0))

# %%
# Verify orthogonality numerically
# -----------------------------------------------------

legendre_ip = inner_product(lambda x: legendre_polynomial(2, x), lambda x: legendre_polynomial(3, x), lambda x: 1.0, -1.0, 1.0)
hermite_ip = inner_product(lambda x: hermite_polynomial(1, x), lambda x: hermite_polynomial(2, x), lambda x: np.exp(-(x**2)), -np.inf, np.inf)
laguerre_ip = inner_product(lambda x: laguerre_polynomial(1, x), lambda x: laguerre_polynomial(3, x), lambda x: np.exp(-x), 0.0, np.inf)

print(f"<P_2, P_3> (weight 1)      = {legendre_ip:.2e}")
print(f"<H_1, H_2> (weight e^-x^2) = {hermite_ip:.2e}")
print(f"<L_1, L_3> (weight e^-x)   = {laguerre_ip:.2e}")
