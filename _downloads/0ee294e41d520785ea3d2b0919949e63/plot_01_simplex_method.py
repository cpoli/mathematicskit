r"""
Dantzig's simplex method: walking the vertices of a production LP
=======================================================================

Maximize the profit :math:`3x + 5y` from two products subject to
:math:`x + 2y \leq 40` labor-hours and :math:`2x + y \leq 30` units of
material. The simplex method starts at the vertex :math:`(0, 0)` of the
feasible polygon and pivots along its edges to an adjacent vertex with
a higher profit, until no neighbour improves. The tableau pivots are
written out below, and the answer is checked against
:func:`~mathematicskit.optimization.systems.linear_programming.linear_program`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import linear_program

c = np.array([3.0, 5.0])
a_ub = np.array([[1.0, 2.0], [2.0, 1.0]])
b_ub = np.array([40.0, 30.0])

# %%
# The simplex tableau, pivot by pivot
# -----------------------------------
# Slack variables turn each inequality into an equation, and the slack
# basis is the starting vertex :math:`(0, 0)`. Dantzig's rule brings in
# the variable with the largest reduced profit; the ratio test picks the
# constraint that becomes tight first, which fixes the next vertex.

m, n = a_ub.shape
tableau = np.zeros((m + 1, n + m + 1))
tableau[:m, :n] = a_ub
tableau[:m, n : n + m] = np.eye(m)
tableau[:m, -1] = b_ub
tableau[-1, :n] = -c
basis = list(range(n, n + m))


def current_vertex():
    x = np.zeros(n + m)
    x[basis] = tableau[:m, -1]
    return x[:n]


vertices = [current_vertex()]
while np.min(tableau[-1, :-1]) < -1e-12:
    col = int(np.argmin(tableau[-1, :-1]))
    positive = tableau[:m, col] > 1e-12
    ratios = np.where(positive, tableau[:m, -1] / np.where(positive, tableau[:m, col], 1.0), np.inf)
    row = int(np.argmin(ratios))
    tableau[row] /= tableau[row, col]
    for r in range(m + 1):
        if r != row:
            tableau[r] -= tableau[r, col] * tableau[row]
    basis[row] = col
    vertices.append(current_vertex())

vertices = np.array(vertices)
for k, v in enumerate(vertices):
    print(f"vertex {k}: (x, y) = ({v[0]:.3f}, {v[1]:.3f}), profit = {c @ v:.3f}")

result = linear_program(-c, a_ub=a_ub, b_ub=b_ub)  # linprog minimizes, so negate
print(f"linear_program optimum: {result.x.round(3)}, profit = {-result.fun:.3f}")

# %%
# The path along the feasible polygon
# -----------------------------------

polygon = np.array([[0.0, 0.0], [15.0, 0.0], [20.0 / 3.0, 50.0 / 3.0], [0.0, 20.0]])
fig, ax = plt.subplots(figsize=(6, 5))
ax.fill(*polygon.T, color="0.9", label="feasible region")
xs = np.linspace(0.0, 25.0, 2)
ax.plot(xs, (40.0 - xs) / 2.0, color="0.5", lw=1)
ax.plot(xs, 30.0 - 2.0 * xs, color="0.5", lw=1)
for level in (50.0, 100.0, c @ vertices[-1]):
    ax.plot(xs, (level - 3.0 * xs) / 5.0, ":", color="tab:green", lw=1)
ax.plot(*polygon.T, "ko", ms=5)
ax.plot(*vertices.T, "o-", color="tab:blue", lw=2.5, ms=8, label="simplex path")
for k, v in enumerate(vertices):
    ax.annotate(f"{k}: profit {c @ v:.1f}", v, textcoords="offset points", xytext=(8, 4))
ax.set_xlim(-1.0, 25.0)
ax.set_ylim(-1.0, 25.0)
ax.set_xlabel("units of A (x)")
ax.set_ylabel("units of B (y)")
ax.legend(loc="upper right")
ax.set_title("Simplex method: vertex to adjacent vertex (Dantzig, 1947)")
