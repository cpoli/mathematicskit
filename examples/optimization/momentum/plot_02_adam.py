r"""
Adam on a badly scaled problem
====================================

A quadratic whose two coordinates differ in curvature by a factor of
1000. Fixed-step gradient descent must use a step small enough for the
stiff coordinate, so it crawls along the flat one. Adam rescales each
coordinate by its root-mean-square gradient and moves both at a similar
speed.
"""

# %%
import numpy as np

from mathematicskit.optimization import Adam, GradientDescent
from mathematicskit.optimization.visualizers.plots import plot_contour_path

# %%
# f(x, y) = x^2 / 1000 + y^2
# -----------------------------------------------------

f = lambda x: x[0] ** 2 / 1000.0 + x[1] ** 2
grad = lambda x: np.array([2.0 * x[0] / 1000.0, 2.0 * x[1]])
x0 = np.array([-1.5, 2.5])

gd = GradientDescent(alpha=0.4, max_iter=300).minimize(f, grad, x0)
adam = Adam(alpha=0.05, max_iter=300).minimize(f, grad, x0)
print(f"after 300 steps: gradient descent x = {gd.x.round(4)}, Adam x = {adam.x.round(4) + 0.0}")

ax = plot_contour_path(f, gd, x_range=(-2.0, 0.5), y_range=(-1.0, 3.0), label="gradient descent")
plot_contour_path(f, adam, ax=ax, x_range=(-2.0, 0.5), y_range=(-1.0, 3.0), label="Adam (Kingma & Ba, 2014)")
