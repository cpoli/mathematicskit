r"""
The argument principle: counting zeros minus poles by winding
=============================================================

:math:`\frac{1}{2\pi i}\oint_\gamma f'/f = N - P`, the number of zeros
minus poles inside :math:`\gamma`. Equivalently, the image curve
:math:`f(\gamma)` winds :math:`N - P` times around 0. For a quintic,
counting roots inside disks of growing radius needs no root finder at
all.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import argument_principle, circle_contour

coefficients = [1, -0.5, 0.2, -1.3, 0.1, 0.4]
p = lambda z: np.polyval(coefficients, z)
dp = lambda z: np.polyval(np.polyder(coefficients), z)
roots = np.roots(coefficients)

# %%
# Root counts from the argument principle
# ---------------------------------------

for radius in (0.3, 0.6, 0.9, 1.2, 1.5):
    circle = circle_contour(0.0, radius)
    print(
        f"|z| < {radius}: phase winding = {argument_principle(p, circle)}, "
        f"oint p'/p = {argument_principle(p, circle, fprime=dp)}, "
        f"numpy.roots = {int(np.sum(np.abs(roots) < radius))}"
    )

# %%
# The image curve winds around 0 once per enclosed zero
# -----------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for radius in (0.6, 1.2):
    circle = circle_contour(0.0, radius)
    z = circle.points()
    axes[0].plot(z.real, z.imag, label=f"|z| = {radius}")
    w = p(z)
    axes[1].plot(w.real, w.imag, label=f"p(|z| = {radius}): winds {argument_principle(p, circle)}x")
axes[0].plot(roots.real, roots.imag, "ko", label="roots")
axes[1].plot(0, 0, "k+", ms=12)
for ax, title in zip(axes, ("z-plane", "w = p(z)"), strict=False):
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(fontsize=8)
