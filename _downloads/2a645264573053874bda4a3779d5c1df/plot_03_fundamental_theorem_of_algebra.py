r"""
The fundamental theorem of algebra, by winding numbers
======================================================

Gauss's 1799 dissertation gave the first substantial proof that every
polynomial of degree :math:`n` has :math:`n` complex roots. The modern
topological proof counts windings: on a huge circle
:math:`p(z) \approx z^n` winds :math:`n` times around 0, on a tiny
circle about a non-root it winds 0 times, so as the circle grows the
image curve must pass through 0 -- and the argument principle says it
does so exactly :math:`n` times.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import argument_principle, circle_contour

coefficients = [1.0, -2.0, 3.0, 1.0, -4.0, 2.0]  # degree 5
p = np.poly1d(coefficients)
print("roots:", np.round(np.roots(coefficients), 4))

# %%
# Winding number of :math:`p(|z| = R)` about 0 as R grows
# -------------------------------------------------------

radii = [0.2, 0.6, 1.0, 1.4, 2.0, 5.0]
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, R in zip(axes.ravel(), radii, strict=False):
    contour = circle_contour(0.0, R)
    image = p(contour.points(2000))
    winding = argument_principle(p, contour)
    inside = int(np.sum(np.abs(np.roots(coefficients)) < R))
    print(f"R = {R:3.1f}: winding number {winding}, roots inside {inside}")
    ax.plot(image.real, image.imag, lw=1)
    ax.plot(0, 0, "r+", ms=12, mew=2)
    ax.set_title(f"R = {R}: winds {winding} times")
    ax.set_aspect("equal")
fig.suptitle("Image of |z| = R under a degree-5 polynomial")
fig.tight_layout()

plt.show()
