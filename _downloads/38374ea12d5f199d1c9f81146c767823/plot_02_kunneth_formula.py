r"""
The Künneth formula for products (1923)
=======================================

Hermann Künneth showed that the Betti numbers of a product are a
convolution:
:math:`\beta_k(X \times Y) = \sum_{i+j=k} \beta_i(X)\,\beta_j(Y)`,
the coefficients of the product of the two Poincaré polynomials. The
product of two simplicial complexes is triangulated by staircase paths
through each product of simplices. Circle times circle is the torus,
:math:`(1 + t)^2 = 1 + 2t + t^2`, and circle times sphere gives
:math:`(1 + t)(1 + t^2)`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import betti_numbers, circle, simplex, simplicial_product, sphere


def poincare_polynomial(betti):
    return " + ".join(f"{b}t^{k}" if k else f"{b}" for k, b in enumerate(betti) if b) or "0"


pairs = {
    "circle x circle": (circle(3), circle(4)),
    "circle x interval": (circle(4), simplex(1)),
    "circle x sphere": (circle(3), sphere(2)),
    "sphere x sphere": (sphere(2), sphere(2)),
    "torus x circle": (simplicial_product(circle(3), circle(3)), circle(3)),
}
for name, (X, Y) in pairs.items():
    product = simplicial_product(X, Y)
    predicted = poincare_polynomial(np.convolve(betti_numbers(X), betti_numbers(Y)))
    print(f"{name:>17}: predicted {predicted:>22}, computed {poincare_polynomial(betti_numbers(product)):>22}, f = {product.f_vector}")

# %%
# A square times a square, drawn as a flat torus
# ----------------------------------------------
#
# Each square of the grid is a product of two edges, cut into two
# triangles along the staircase diagonal. The squares that wrap around
# (glued to the opposite side) are left out of the drawing.

T = simplicial_product(circle(4), circle(4))
grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
fig, ax = plt.subplots(figsize=(5, 5))
for tri in T.simplices(2):
    pts = grid[list(tri)]
    if np.ptp(pts[:, 0]) <= 1 and np.ptp(pts[:, 1]) <= 1:
        ax.add_patch(plt.Polygon(pts, alpha=0.4, ec="k"))
ax.set_xlim(-0.2, 3.2)
ax.set_ylim(-0.2, 3.2)
ax.set_aspect("equal")
ax.set_title(f"$S^1 \\times S^1$: f = {T.f_vector}, β = {betti_numbers(T)}")
