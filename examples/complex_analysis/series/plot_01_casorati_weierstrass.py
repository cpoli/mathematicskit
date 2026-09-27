r"""
The Casorati-Weierstrass theorem: near an essential singularity f comes close to every value
============================================================================================

An isolated singularity is removable, a pole, or *essential*, depending
on whether the Laurent series has no negative powers, finitely many, or
infinitely many. Casorati and Weierstrass showed that near an essential
singularity the values of :math:`f` are dense in the plane. For
:math:`e^{1/z}` at 0 we find a point within :math:`\delta = 10^{-3}` of
0 where :math:`f` hits an arbitrary target :math:`w`, and the phase
portrait shows every color packed into every neighbourhood of 0.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import domain_coloring, laurent_coefficients
from mathematicskit.complex_analysis.visualizers import plot_domain_coloring

f = lambda z: np.exp(1 / z)

# %%
# Infinitely many negative powers
# -------------------------------

series = laurent_coefficients(f, 0.0, 1.0, 6)
print("a_{-k} for k = 0..6:", np.round([series.coefficient(-k).real for k in range(7)], 6), "(= 1/k!)")

# %%
# Hitting any target value arbitrarily close to 0
# -----------------------------------------------
# e^{1/z} = w has the solutions z = 1 / (log w + 2 pi i n); large n makes |z| small.

delta = 1e-3
for w in (5.0, -2 + 1j, 1e-4j):
    n = int(np.ceil(1 / (2 * np.pi * delta))) + 1
    z = 1 / (np.log(w) + 2j * np.pi * n)
    print(f"target w = {w}: z = {z:.2e}, |z| = {abs(z):.1e} < {delta}, f(z) = {f(z):.6f}")

# %%
# Zooming in on the singularity
# -----------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, half_width in zip(axes, (1.0, 0.1, 0.01), strict=False):
    result = domain_coloring(f, (-half_width, half_width), (-half_width, half_width), resolution=400)
    plot_domain_coloring(result, ax=ax, title=rf"$e^{{1/z}}$, $|{{\rm Re}}\,z|, |{{\rm Im}}\,z| < {half_width}$")
fig.tight_layout()
