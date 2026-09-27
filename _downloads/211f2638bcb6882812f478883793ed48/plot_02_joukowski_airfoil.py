r"""
The Joukowski map: turning circles into airfoils
================================================

:math:`J(z) = z + 1/z` flattens the unit circle onto :math:`[-2, 2]`.
A circle through :math:`z = 1` that encloses :math:`-1` maps to an
airfoil with a sharp trailing edge, where :math:`J'(1) = 0` and the map
stops being conformal. Joukowski used this in 1910 to carry the
solvable flow around a cylinder over to a wing profile.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.complex_analysis import circle_contour, complex_derivative, joukowski_map

# %%
# Circles and their images
# ------------------------

centers = {"unit circle": 0.0, "shifted left": -0.1, "shifted left and up (airfoil)": -0.1 + 0.15j}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for name, c in centers.items():
    circle = circle_contour(c, abs(1 - c))  # every circle passes through z = 1
    z = circle.points(600)
    w = joukowski_map(z)
    axes[0].plot(z.real, z.imag, label=name)
    axes[1].plot(w.real, w.imag, label=name)
axes[0].plot([1, -1], [0, 0], "kx")
for ax, title in zip(axes, ("z-plane", "w = z + 1/z"), strict=False):
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(fontsize=8)
print("J'(1) =", complex_derivative(joukowski_map, 1.0), "(the trailing edge is a critical point)")
