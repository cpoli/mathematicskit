r"""
Brouwer's fixed-point theorem (1911)
====================================

Every continuous map of a disc (or triangle) to itself has a fixed
point. The proof here is constructive. Label each vertex of a fine
triangulation with a coordinate the map does not increase. That
labelling obeys Sperner's rule, so some small triangle carries all three
labels, and on it the map nearly fixes every coordinate. Refining the
grid closes in on a fixed point, which matches a Newton-type solve with
:func:`scipy.optimize.fsolve`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

from mathematicskit.topology import brouwer_fixed_point, triangle_grid


def f(x):
    """A nonlinear swirl of the triangle onto itself (barycentric coordinates in and out)."""
    a, b, c = x
    y = np.array([0.15 + 0.6 * b + 0.2 * a * c, 0.1 + 0.7 * c**2 + 0.1 * a, 0.0])
    y = np.clip(y, 0, None)
    y[2] = max(0.0, 1 - y[0] - y[1])
    return y / y.sum()


exact = fsolve(lambda z: f(np.array([z[0], z[1], 1 - z[0] - z[1]]))[:2] - z, [0.3, 0.3])
exact = np.array([*exact, 1 - exact.sum()])
print("fixed point by fsolve:", np.round(exact, 6))
for n in (4, 16, 64, 256):
    result = brouwer_fixed_point(f, n)
    error = np.max(np.abs(result.point - exact))
    print(f"n = {n:3d}: Sperner triangle centroid {np.round(result.point, 4)}, error {error:.2e}, residual {result.residual:.2e}")

# %%
# The labels and the fully labelled triangle
# ------------------------------------------

n = 24
grid = triangle_grid(n)
x = grid.points / n
labels = np.argmax((np.array([f(p) for p in x]) <= x + 1e-15) & (x > 0), axis=1)
result = brouwer_fixed_point(f, n)
corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.triplot(*grid.cartesian.T, grid.triangles, color="0.8", lw=0.5)
for k, color in enumerate(("tab:red", "tab:green", "tab:blue")):
    ax.scatter(*grid.cartesian[labels == k].T, s=14, color=color, label=f"label {k}: f decreases coordinate {k}")
ax.add_patch(plt.Polygon(result.triangle @ corners, color="gold", ec="k", zorder=3))
ax.scatter(*(exact @ corners), marker="*", s=200, color="k", zorder=4, label="true fixed point")
ax.set_aspect("equal")
ax.axis("off")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Brouwer: a fully labelled triangle traps the fixed point")
