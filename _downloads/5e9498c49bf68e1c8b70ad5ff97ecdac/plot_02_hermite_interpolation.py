r"""
Hermite interpolation: matching slopes as well as values
===========================================================

Charles Hermite's 1878 interpolant matches both :math:`f(x_i)` and
:math:`f'(x_i)`, so :math:`n + 1` nodes fix a polynomial of degree
:math:`2n + 1`. On the same few nodes it follows :math:`\sin` far more
closely than the Lagrange interpolant, which knows only the values.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import HermiteInterpolant, LagrangeInterpolant

# %%
# Four nodes on one period of :math:`\sin x`
# ----------------------------------------------

x = np.linspace(0.0, 2.0 * np.pi, 4)
hermite = HermiteInterpolant(x, np.sin(x), dydx=np.cos(x))
lagrange = LagrangeInterpolant(x, np.sin(x))

t = np.linspace(0.0, 2.0 * np.pi, 400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.plot(t, np.sin(t), "k", lw=2, label=r"$\sin x$")
ax1.plot(t, hermite(t), "--", color="firebrick", label="Hermite (degree 7)")
ax1.plot(t, lagrange(t), ":", color="steelblue", label="Lagrange (degree 3)")
ax1.scatter(x, np.sin(x), color="black", zorder=3)
for xi in x:
    ax1.plot([xi - 0.4, xi + 0.4], [np.sin(xi) - 0.4 * np.cos(xi), np.sin(xi) + 0.4 * np.cos(xi)], color="gray", lw=1)
ax1.set_title("Values and slopes at the nodes")
ax1.legend(fontsize=8)

ax2.semilogy(t, np.abs(hermite(t) - np.sin(t)) + 1e-17, color="firebrick", label="Hermite")
ax2.semilogy(t, np.abs(lagrange(t) - np.sin(t)) + 1e-17, color="steelblue", label="Lagrange")
ax2.set_title("Absolute error")
ax2.legend(fontsize=8)
fig.tight_layout()

print(f"max error: Hermite {np.max(np.abs(hermite(t) - np.sin(t))):.2e}, Lagrange {np.max(np.abs(lagrange(t) - np.sin(t))):.2e}")

plt.show()
