r"""
Gauss's arithmetic-geometric mean and the elliptic integral K
===================================================================

Iterates the AGM by hand to show its quadratic convergence, then checks
Gauss's 1799 identity K(m) = pi / (2 AGM(1, sqrt(1 - m))) and uses E(m)
for the perimeter of an ellipse.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import arithmetic_geometric_mean, complete_elliptic_integral_first_kind, complete_elliptic_integral_second_kind

# %%
# Quadratic convergence: the number of correct digits doubles each step
# ----------------------------------------------------------------------

a, b = 1.0, math.sqrt(2.0)
limit = arithmetic_geometric_mean(a, b)
for step in range(5):
    print(f"step {step}: a = {a:.16f}, b = {b:.16f}, |a - AGM| = {abs(a - limit):.1e}")
    a, b = (a + b) / 2, math.sqrt(a * b)
print(f"Gauss's constant 1/AGM(1, sqrt 2) = {1 / limit:.12f}")

# %%
# K(m) from the AGM
# -----------------------------------------------------

m = np.linspace(0.0, 0.999, 200)
k_agm = np.pi / (2 * arithmetic_geometric_mean(1.0, np.sqrt(1 - m)))
print(f"\nmax |K(m) - pi/(2 AGM)| = {np.max(np.abs(complete_elliptic_integral_first_kind(m) - k_agm)):.1e}")

fig, ax = plt.subplots()
ax.plot(m, complete_elliptic_integral_first_kind(m), label="K(m)")
ax.plot(m, complete_elliptic_integral_second_kind(m), label="E(m)")
ax.axhline(np.pi / 2, color="gray", ls=":", lw=1)
ax.set_xlabel("parameter m")
ax.set_title("Complete elliptic integrals")
ax.legend()

# %%
# The perimeter of an ellipse is 4 a E(e^2)
# -----------------------------------------------------

semi_major, semi_minor = 2.0, 1.0
ecc2 = 1 - (semi_minor / semi_major) ** 2
perimeter = 4 * semi_major * complete_elliptic_integral_second_kind(ecc2)
t = np.linspace(0, 2 * np.pi, 200001)
polyline = np.sum(np.hypot(np.diff(semi_major * np.cos(t)), np.diff(semi_minor * np.sin(t))))
print(f"\nellipse 2 x 1: 4aE(e^2) = {perimeter:.10f}, polyline length = {polyline:.10f}")
