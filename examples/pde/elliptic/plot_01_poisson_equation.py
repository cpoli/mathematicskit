r"""
Poisson's equation: a potential with sources
============================================

Laplace's equation :math:`\nabla^2 u = 0` governs a potential in empty
space; Poisson added the source term, :math:`\nabla^2 u = f`, for the
potential inside matter. This script solves Poisson's equation on the
unit square for a pair of opposite point-like charges, then verifies the
five-point finite-difference solver's second-order accuracy against a
problem with a known solution.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import solve_poisson_2d
from mathematicskit.pde.visualizers import plot_field_2d


def charges(X, Y):
    blob = lambda x0, y0: np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / 0.002)
    return -200.0 * (blob(0.35, 0.5) - blob(0.65, 0.5))


# %%
# The potential of a dipole in a grounded box
# -------------------------------------------

sol = solve_poisson_2d(charges, n=(81, 81))
print(f"sparse direct solve: {(len(sol.x) - 2) ** 2} unknowns, residual {sol.residual_norm:.1e}")
fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
plot_field_2d(sol, ax=ax1, levels=30, cmap="RdBu_r")
ax1.set_title(r"$\nabla^2 u = f$: two opposite charges, $u = 0$ on the box")
fig1.tight_layout()

# %%
# Second-order convergence
# ------------------------
# For :math:`f = -2\pi^2 \sin(\pi x)\sin(\pi y)` the exact solution is
# :math:`\sin(\pi x)\sin(\pi y)`; halving the grid spacing quarters the error.

ns = np.array([9, 17, 33, 65])
errors = []
for n in ns:
    s = solve_poisson_2d(lambda X, Y: -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y), n=(n, n))
    X, Y = np.meshgrid(s.x, s.y, indexing="ij")
    errors.append(np.max(np.abs(s.u - np.sin(np.pi * X) * np.sin(np.pi * Y))))
h = 1.0 / (ns - 1)
print(f"observed order: {np.polyfit(np.log(h), np.log(errors), 1)[0]:.2f}")
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.loglog(h, errors, "o-", label="five-point Poisson solver")
ax2.loglog(h, errors[0] * (h / h[0]) ** 2, "k--", label=r"$O(h^2)$")
ax2.set_xlabel("$h$")
ax2.set_ylabel("max error")
ax2.legend()
fig2.tight_layout()

plt.show()
