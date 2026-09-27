r"""
Euler's formula: e^{i theta} = cos theta + i sin theta
======================================================

Euler's 1748 formula wraps the real line around the unit circle:
:math:`e^{i\theta}` is the point at angle :math:`\theta`, and
:math:`e^{i\pi} + 1 = 0`. The domain coloring of :math:`e^z` shows the
consequence for the whole plane: the hue (phase) depends only on
:math:`\operatorname{Im} z`, so :math:`e^z` repeats every :math:`2\pi i`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import circle_contour, domain_coloring
from mathematicskit.complex_analysis.visualizers import plot_domain_coloring

# %%
# e^{i theta} traces the unit circle
# ----------------------------------

theta = np.linspace(0, 2 * np.pi, 9)
unit_circle = circle_contour(0.0, 1.0)
print("max |e^{i theta} - (cos theta + i sin theta)| =", np.max(np.abs(np.exp(1j * theta) - (np.cos(theta) + 1j * np.sin(theta)))))
print("e^{i pi} + 1 =", np.exp(1j * np.pi) + 1)

fig, ax = plt.subplots()
circle = unit_circle.points()
ax.plot(circle.real, circle.imag, "C0")
for t in theta[:-1]:
    w = np.exp(1j * t)
    ax.plot([0, w.real], [0, w.imag], "C1", lw=0.8)
    ax.annotate(rf"$\theta={t / np.pi:.2g}\pi$", (w.real, w.imag), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_aspect("equal")
ax.set_title(r"$e^{i\theta} = \cos\theta + i\sin\theta$")

# %%
# e^z is periodic with period 2 pi i
# ----------------------------------

result = domain_coloring(np.exp, (-2, 2), (-2 * np.pi, 2 * np.pi), resolution=300)
plot_domain_coloring(result, title=r"$e^z$: phase depends only on Im z")
print("e^{z + 2 pi i} == e^z:", np.allclose(np.exp(result.z + 2j * np.pi), result.w))
