r"""
Runge's phenomenon: equally spaced interpolation diverges
=========================================================

Carl Runge's 1901 example :math:`f(x) = 1/(1+25x^2)` is smooth on
:math:`[-1, 1]`, yet interpolating it at *more* equally spaced nodes
makes the fit *worse*: oscillations near the endpoints grow without
bound as the degree increases. This script shows the overshoot for
growing degree, the maximum error blowing up, and the exponentially
growing Lebesgue constant of equally spaced nodes that causes it.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import LagrangeInterpolant, runge_function
from mathematicskit.numerical_analysis.systems.chebyshev import runge_phenomenon_errors
from mathematicskit.numerical_analysis.utils.error_analysis import lebesgue_constant

# %%
# More equally spaced nodes, bigger wiggles
# -----------------------------------------

x_fine = np.linspace(-1.0, 1.0, 600)
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(x_fine, runge_function(x_fine), color="black", lw=2, label=r"$f(x) = 1/(1+25x^2)$")
for n, color in ((6, "goldenrod"), (11, "darkorange"), (16, "firebrick")):
    nodes = np.linspace(-1.0, 1.0, n)
    p = LagrangeInterpolant(nodes, runge_function(nodes))
    ax1.plot(x_fine, p.evaluate(x_fine), color=color, label=f"degree {n - 1}, equally spaced")
    ax1.plot(nodes, runge_function(nodes), "o", ms=3, color=color)
ax1.set_ylim(-2.5, 2.5)
ax1.set_xlabel("$x$")
ax1.set_title("Runge's phenomenon: oscillations grow near the endpoints")
ax1.legend(fontsize=8)
fig1.tight_layout()

# %%
# The maximum error diverges with degree
# --------------------------------------

degrees = [5, 10, 15, 20, 25, 30]
equal_errors, _ = runge_phenomenon_errors(degrees)
for d, e in zip(degrees, equal_errors):
    print(f"degree={d:2d}  equally spaced max |error| = {e:.3e}")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.semilogy(degrees, equal_errors, "o-", color="firebrick")
ax2.set_xlabel("polynomial degree")
ax2.set_ylabel(r"$\max |f - p_n|$")
ax2.set_title("Equally spaced interpolation error grows without bound")
fig2.tight_layout()

# %%
# The cause: an exponentially growing Lebesgue constant
# -----------------------------------------------------

for n in (5, 10, 20, 30):
    print(f"n={n:2d} nodes  Lebesgue constant = {lebesgue_constant(np.linspace(-1.0, 1.0, n)):.2e}")

plt.show()
