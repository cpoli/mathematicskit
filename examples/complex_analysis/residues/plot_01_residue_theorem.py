r"""
Cauchy's residue theorem: the contour integral is 2 pi i times the enclosed residues
====================================================================================

For :math:`f` meromorphic inside :math:`\gamma`,
:math:`\oint_\gamma f = 2\pi i \sum_k n(\gamma, z_k)\operatorname{Res}(f, z_k)`.
Growing a circle past the poles of
:math:`f(z) = e^{z}/((z - 1)(z + 2)(z - 3i))` adds each residue in
turn, and the same theorem evaluates the real integral
:math:`\int_{-\infty}^\infty dx/(1 + x^4) = \pi/\sqrt 2`.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from mathematicskit.complex_analysis import circle_contour, residue, residue_theorem
from mathematicskit.complex_analysis.visualizers import plot_contour

poles = [1, -2, 3j]
f = lambda z: np.exp(z) / ((z - 1) * (z + 2) * (z - 3j))

# %%
# Growing circles pick up one residue at a time
# ---------------------------------------------

fig, ax = plt.subplots()
for radius in (0.5, 1.5, 2.5, 3.5):
    circle = circle_contour(0.0, radius)
    result = residue_theorem(f, circle, poles)
    plot_contour(circle, marked_points=poles, ax=ax)
    print(f"r={radius}: enclosed={result.winding_numbers.tolist()}  oint f = {result.integral:.8f}  2 pi i sum Res = {result.predicted:.8f}")
ax.set_title("Circles of radius 0.5, 1.5, 2.5, 3.5 around three poles")

# %%
# A real integral by residues
# ---------------------------
# Closing the real line with a large upper semicircle encloses the poles
# e^{i pi/4} and e^{3 i pi/4} of 1/(1 + z^4); the arc's contribution vanishes.

g = lambda z: 1 / (1 + z**4)
by_residues = 2j * math.pi * sum(residue(g, np.exp(1j * math.pi * k / 4)) for k in (1, 3))
by_quadrature, _ = integrate.quad(lambda x: 1 / (1 + x**4), -np.inf, np.inf)
print(f"residues: {by_residues.real:.12f}   quad: {by_quadrature:.12f}   pi/sqrt(2): {math.pi / math.sqrt(2):.12f}")
