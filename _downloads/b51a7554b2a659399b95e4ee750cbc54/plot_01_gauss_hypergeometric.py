r"""
Gauss's hypergeometric function contains the elementary functions
======================================================================

Recovers log(1 + z), arcsin z, and a Chebyshev polynomial from 2F1 with
special parameters, and checks Gauss's summation theorem at z = 1.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import chebyshev_polynomial, gamma_function, hypergeometric_2f1

# %%
# Three familiar functions in one family
# -----------------------------------------------------

z = np.linspace(-0.95, 0.95, 300)
log_curve = z * hypergeometric_2f1(1.0, 1.0, 2.0, -z)
arcsin_curve = z * hypergeometric_2f1(0.5, 0.5, 1.5, z**2)
cheb_curve = hypergeometric_2f1(-4, 4, 0.5, (1 - z) / 2)

fig, ax = plt.subplots()
ax.plot(z, log_curve, label=r"$z\,{}_2F_1(1,1;2;-z) = \ln(1+z)$")
ax.plot(z, arcsin_curve, label=r"$z\,{}_2F_1(\frac{1}{2},\frac{1}{2};\frac{3}{2};z^2) = \arcsin z$")
ax.plot(z, cheb_curve, label=r"${}_2F_1(-4,4;\frac{1}{2};\frac{1-z}{2}) = T_4(z)$")
ax.set_xlabel("z")
ax.legend(fontsize="small")
ax.set_title("Special cases of the hypergeometric function")

print(f"max |log error|    = {np.max(np.abs(log_curve - np.log1p(z))):.1e}")
print(f"max |arcsin error| = {np.max(np.abs(arcsin_curve - np.arcsin(z))):.1e}")
print(f"max |T_4 error|    = {np.max(np.abs(cheb_curve - chebyshev_polynomial(4, z))):.1e}")

# %%
# Gauss's summation theorem at z = 1
# -----------------------------------------------------

a, b, c = 0.5, 0.25, 2.0
gauss = gamma_function(c) * gamma_function(c - a - b) / (gamma_function(c - a) * gamma_function(c - b))
print(f"\n2F1({a}, {b}; {c}; 1) = {hypergeometric_2f1(a, b, c, 1.0):.12f}")
print(f"Gamma ratio          = {gauss:.12f}")
