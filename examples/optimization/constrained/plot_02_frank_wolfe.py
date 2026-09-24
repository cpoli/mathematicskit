r"""
Frank-Wolfe: optimization over the probability simplex
============================================================

Minimizes a convex quadratic over the triangle
:math:`\{x \geq 0,\ x_1 + x_2 + x_3 = 1\}` without ever projecting onto
it. Each step solves a linear program, whose solution is a vertex, and
moves toward that vertex. The Frank-Wolfe gap certifies how far the
objective is from optimal.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import frank_wolfe

# %%
# Project a point onto the simplex
# --------------------------------

y = np.array([1.0, 0.2, -0.5])
f = lambda x: float(np.sum((x - y) ** 2))
grad = lambda x: 2.0 * (x - y)
result = frank_wolfe(f, grad, [0.0, 0.0, 1.0], a_eq=np.ones((1, 3)), b_eq=np.array([1.0]), max_iter=500)
print(f"Frank-Wolfe solution: {result.x.round(4)} (exact projection: [0.9, 0.1, 0.0])")

# %%
# Iterates in barycentric coordinates, and the duality gap
# --------------------------------------------------------

corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
xy = result.path @ corners
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.fill(*corners.T, color="0.92")
ax1.plot(*xy.T, ".-", ms=3, lw=0.8)
ax1.plot(*(np.array([0.9, 0.1, 0.0]) @ corners), "*", ms=12, color="tab:red", label="optimum")
ax1.set_aspect("equal")
ax1.axis("off")
ax1.legend()
ax1.set_title("Iterates on the simplex")
k = np.arange(len(result.extra["gaps"]))
ax2.loglog(k + 1, result.extra["gaps"], label="Frank-Wolfe gap")
f_star = f(np.array([0.9, 0.1, 0.0]))
ax2.loglog(k + 1, [f(x) - f_star for x in result.path[: len(k)]], label=r"$f(x_k) - f^*$")
ax2.set_xlabel("iteration k + 1")
ax2.legend()
ax2.set_title("O(1/k) convergence (Frank & Wolfe, 1956)")
fig.tight_layout()
