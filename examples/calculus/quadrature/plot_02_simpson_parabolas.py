r"""
Simpson's rule: parabolas through pairs of panels
=======================================================

Simpson's rule integrates cubics exactly and converges at fourth order.
This example compares it with the trapezoidal rule on a smooth
integrand and shows the parabolic arcs it fits.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from mathematicskit.calculus import SimpsonsRule, TrapezoidalRule
from mathematicskit.calculus.visualizers.plots import plot_quadrature_convergence

# %%
# Exact for cubics
# -----------------------------------------------------

cubic = SimpsonsRule(n=2).integrate(lambda x: x**3 - 2 * x + 1, 0.0, 2.0)
print(f"Simpson with 2 panels on x^3 - 2x + 1 over [0, 2]: {cubic.value} (exact 2)")

# %%
# The parabolic arcs
# -----------------------------------------------------

f, a, b, n = (lambda x: np.exp(-x) * np.sin(3 * x) + 1), 0.0, 3.0, 6
xs = np.linspace(a, b, n + 1)
fig, ax = plt.subplots()
grid = np.linspace(a, b, 400)
ax.plot(grid, f(grid), "k", label="f")
for i in range(0, n, 2):
    seg = xs[i : i + 3]
    coeffs = np.polyfit(seg, f(seg), 2)
    local = np.linspace(seg[0], seg[-1], 50)
    ax.fill_between(local, np.polyval(coeffs, local), alpha=0.3)
ax.plot(xs, f(xs), "o")
ax.set_title("Simpson's rule: one parabola per pair of panels")
ax.legend()

# %%
# Fourth-order convergence
# -----------------------------------------------------

exact, _ = quad(f, a, b, epsabs=1e-14)
ns = [2, 4, 8, 16, 32, 64, 128]
ax = plot_quadrature_convergence(lambda n: TrapezoidalRule(n), f, a, b, exact, ns, label="trapezoidal O(h^2)")
plot_quadrature_convergence(lambda n: SimpsonsRule(n), f, a, b, exact, ns, ax=ax, label="Simpson O(h^4)")
