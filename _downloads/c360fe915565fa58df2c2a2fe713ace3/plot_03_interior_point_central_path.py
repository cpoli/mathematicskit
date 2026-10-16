r"""
Karmarkar's interior-point method: the central path
===================================================

The simplex method walks the edges of the feasible polygon. Karmarkar
(1984) cut through its interior instead, with an algorithm that was
both polynomial-time and fast in practice. Its descendants, the
primal-dual path-following methods used by every modern LP solver,
track the *central path*: the points where every product
:math:`x_i s_i` of a primal variable and its dual slack equals the same
:math:`\mu`. As :math:`\mu \to 0` the path ends at the optimal vertex.

The problem is the production LP of the simplex example: maximize
:math:`3x + 5y` subject to :math:`x + 2y \leq 40`, :math:`2x + y \leq 30`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import interior_point_lp, linear_program

c = np.array([-3.0, -5.0])  # minimize the negated profit
a_ub = np.array([[1.0, 2.0], [2.0, 1.0]])
b_ub = np.array([40.0, 30.0])

# %%
# Interior-point iterates
# -------------------------

result = interior_point_lp(c, a_ub, b_ub, sigma=0.3, eta=0.9)
for k, (x, mu) in enumerate(zip(result.path, result.extra["duality_measure"], strict=False)):
    if k < 6 or k == len(result.path) - 1:
        print(f"iterate {k:2d}: (x, y) = ({x[0]:7.3f}, {x[1]:7.3f}), profit {-c @ x:8.3f}, mu = {mu:.1e}")
print(f"converged in {result.iterations} iterations to {result.x.round(6)}")

for method in ("highs-ipm", "highs-ds"):
    check = linear_program(c, a_ub=a_ub, b_ub=b_ub, method=method)
    print(f"linprog {method:9s}: {check.x.round(6)}, profit {-check.fun:.6f}")

# %%
# Centring: how close the iterates stay to the central path
# -----------------------------------------------------------
#
# A small centring parameter :math:`\sigma` aims straight at the optimum
# and hugs the boundary; a larger one follows the central path through
# the middle of the polygon, at the cost of more iterations. The simplex
# path from the earlier example, :math:`(0,0) \to (0,20) \to
# (20/3, 50/3)`, runs along the edges.

polygon = np.array([[0.0, 0.0], [15.0, 0.0], [20.0 / 3.0, 50.0 / 3.0], [0.0, 20.0]])
simplex_path = np.array([[0.0, 0.0], [0.0, 20.0], [20.0 / 3.0, 50.0 / 3.0]])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))
ax0.fill(*polygon.T, color="0.92", label="feasible region")
ax0.plot(*np.vstack([polygon, polygon[:1]]).T, "k-", lw=1)
ax0.plot(*simplex_path.T, "s--", color="0.4", ms=7, lw=2, label="simplex (vertex to vertex)")
for sigma, color in [(0.05, "tab:red"), (0.3, "tab:blue"), (0.7, "tab:green")]:
    run = interior_point_lp(c, a_ub, b_ub, sigma=sigma, eta=0.9)
    ax0.plot(*run.path.T, "o-", color=color, ms=4, lw=1.2, label=rf"interior point, $\sigma = {sigma}$ ({run.iterations} iterations)")
    ax1.semilogy(run.extra["duality_measure"], "o-", color=color, ms=3, label=rf"$\sigma = {sigma}$")
ax0.plot(20 / 3, 50 / 3, "k*", ms=16)
ax0.set_xlim(-1, 18)
ax0.set_ylim(-1, 22)
ax0.set_xlabel("units of A (x)")
ax0.set_ylabel("units of B (y)")
ax0.set_title("Through the interior vs. along the edges")
ax0.legend(fontsize=8, loc="upper right")

ax1.set_xlabel("iteration")
ax1.set_ylabel(r"duality measure $\mu = x^T s / N$")
ax1.set_title(r"$\mu$ falls geometrically once the iterates are feasible")
ax1.legend()
fig.tight_layout()

plt.show()
