r"""
Cauchy's integral theorem: the integral of a holomorphic function around a closed contour is zero
=================================================================================================

For :math:`f` holomorphic inside and on a closed contour
:math:`\gamma`, :math:`\oint_\gamma f(z)\,dz = 0`, whatever the shape of
:math:`\gamma`. :math:`1/z` shows what happens when the hypothesis
fails: its integral is :math:`2\pi i` times the number of times
:math:`\gamma` winds around the singularity at 0.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import circle_contour, contour_integral, polygon_contour, winding_number
from mathematicskit.complex_analysis.visualizers import plot_contour

# %%
# Three contours, two integrands
# ------------------------------

contours = {
    "unit circle": circle_contour(0.0, 1.0),
    "star around 0": polygon_contour([1.5 * np.exp(2j * np.pi * k / 10) * (1 if k % 2 == 0 else 0.5) for k in range(10)]),
    "triangle missing 0": polygon_contour([0.5 + 0.5j, 2 + 0.5j, 1 + 2j]),
}
entire = lambda z: np.exp(z) * np.cos(z) + z**4

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, contour) in zip(axes, contours.items(), strict=False):
    plot_contour(contour, marked_points=[0], ax=ax)
    ax.set_title(name)
    print(
        f"{name:20s} oint entire = {contour_integral(entire, contour):.1e}   "
        f"oint 1/z = {contour_integral(lambda z: 1 / z, contour):.6f}   "
        f"winding number about 0 = {winding_number(contour, 0)}"
    )
