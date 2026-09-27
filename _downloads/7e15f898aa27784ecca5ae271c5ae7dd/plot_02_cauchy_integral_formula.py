r"""
Cauchy's integral formula: boundary values determine f and every derivative
===========================================================================

For :math:`f` holomorphic inside a contour :math:`\gamma`,

.. math::

   f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_\gamma \frac{f(z)}{(z - z_0)^{n+1}}\,dz.

Sampling :math:`f` only on the unit circle recovers :math:`e^{z}` and
its derivatives, and :math:`\sin z` everywhere inside the disk.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import cauchy_integral_formula, circle_contour, complex_grid

circle = circle_contour(0.0, 1.0)

# %%
# Derivatives of e^z at z0 from boundary values alone
# ---------------------------------------------------

z0 = 0.3 + 0.2j
for n in range(5):
    value = cauchy_integral_formula(np.exp, circle, z0, n=n)
    print(f"n={n}: formula = {value:.12f}, error vs e^z0 = {abs(value - np.exp(z0)):.1e}")

# %%
# Reconstructing sin z inside the disk
# ------------------------------------

z = complex_grid((-0.9, 0.9), (-0.9, 0.9), 25)
inside = np.abs(z) < 0.9
reconstructed = np.full(z.shape, np.nan, dtype=complex)
reconstructed[inside] = [cauchy_integral_formula(np.sin, circle, p) for p in z[inside]]
error = np.abs(reconstructed - np.sin(z))

fig, ax = plt.subplots()
image = ax.imshow(np.log10(error + 1e-17), origin="lower", extent=(-0.9, 0.9, -0.9, 0.9))
theta = np.linspace(0, 2 * math.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), "k")
fig.colorbar(image, ax=ax, label="log10 |error|")
ax.set_title("sin z inside the disk from its values on the circle")
ax.set_aspect("equal")
