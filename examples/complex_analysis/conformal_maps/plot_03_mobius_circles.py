r"""
Möbius transformations map circles to circles
=============================================

August Ferdinand Möbius's 1855 theory of *Kreisverwandtschaft*
("circle relationship") studied the maps
:math:`w = (az + b)/(cz + d)`, :math:`ad - bc \ne 0`. Every one of them
sends circles and lines to circles and lines, where a line is a circle
through :math:`\infty`. This script maps a family of circles and checks
that each image is again a circle by fitting one to it.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import circle_contour, mobius_transform

a, b, c, d = 1.0, -0.5j, 0.6, 1.0 + 0.3j


def fit_circle(points):
    """Least-squares circle x^2 + y^2 + Dx + Ey + F = 0; returns center, radius, max residual."""
    x, y = points.real, points.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    D, E, F = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)[0]
    center = complex(-D / 2, -E / 2)
    radius = np.sqrt(abs(center) ** 2 - F)
    return center, radius, np.max(np.abs(np.abs(points - center) - radius))


# %%
# Circles in the z-plane and their images
# ---------------------------------------

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
for k, (center, radius) in enumerate([(0.0, 0.5), (0.5 + 0.5j, 0.3), (-0.6, 0.4), (0.2 - 0.6j, 0.25)]):
    z = circle_contour(center, radius).points(400)
    w = mobius_transform(z, a, b, c, d)
    color = f"C{k}"
    ax0.plot(z.real, z.imag, color)
    ax1.plot(w.real, w.imag, color)
    w_center, w_radius, residual = fit_circle(w)
    print(f"circle {k}: image center {w_center:.4f}, radius {w_radius:.4f}, max deviation {residual:.1e}")
for ax, title in ((ax0, "z-plane"), (ax1, "w = (az + b)/(cz + d)")):
    ax.set_aspect("equal")
    ax.set_title(title)

# %%
# A circle through the pole -d/c becomes a line
# ---------------------------------------------

pole = -d / c
theta = np.linspace(0.0, 2.0 * np.pi, 401)[1:-1]  # theta = 0 is the pole itself
z = pole + 0.5 * (1.0 - np.exp(1j * theta))
w = mobius_transform(z, a, b, c, d)
p, q = w[100], w[300]  # two finite image points fix the line
distance = np.abs(((w - p) * np.conj(q - p)).imag) / abs(q - p)
print(f"max distance of the images from the line through two of them: {np.max(distance / np.maximum(1.0, np.abs(w))):.1e} (relative)")

plt.show()
