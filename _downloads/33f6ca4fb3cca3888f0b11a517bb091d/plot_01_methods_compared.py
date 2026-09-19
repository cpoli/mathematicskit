r"""
Root-finding methods and their convergence orders
===================================================

Bisection, Newton-Raphson, secant, and fixed-point iteration all solve
:math:`f(x) = x^2 - 2 = 0`, i.e. find :math:`\sqrt{2}`, but converge at
very different rates: bisection and (generic) fixed-point iteration are
linear (order 1), secant is superlinear (order :math:`\approx 1.618`,
the golden ratio), and Newton-Raphson is quadratic (order 2). This script
runs all four and verifies each empirical order against its theoretical
one via :func:`mathkit.numerical_analysis.utils.error_analysis.estimate_convergence_order`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.numerical_analysis import Bisection, FixedPointIteration, NewtonRaphson, Secant
from mathkit.numerical_analysis.utils.error_analysis import estimate_convergence_order
from mathkit.numerical_analysis.visualizers.plots import plot_convergence_history

# %%
# Solve the same problem four ways
# ---------------------------------

root_exact = np.sqrt(2.0)

result_bisect = Bisection(lambda x: x**2 - 2.0, 0.0, 2.0, tol=1e-13).solve()
result_newton = NewtonRaphson(lambda x: x**2 - 2.0, lambda x: 2.0 * x, x0=1.0, tol=1e-14).solve()
result_secant = Secant(lambda x: x**2 - 2.0, x0=1.0, x1=2.0, tol=1e-14, max_iter=200).solve()
result_fixed = FixedPointIteration(lambda x: 0.5 * (x + 2.0 / x), x0=1.5, tol=1e-14).solve()

for result in (result_bisect, result_newton, result_secant, result_fixed):
    order = estimate_convergence_order(result.history, root_exact)
    print(f"{result.method:>16s}: root={result.root:.10f}, iterations={result.iterations}, empirical order~{order:.2f}")

# %%
# Convergence history, side by side
# -----------------------------------
# Newton's error drops off a cliff (quadratic); bisection is a slow,
# steady straight line on this semilog plot (linear).

fig, ax = plt.subplots(figsize=(6, 4.5))
for result in (result_bisect, result_newton, result_secant, result_fixed):
    plot_convergence_history(result, root_exact=root_exact, ax=ax)
ax.set_title("Convergence history: bisection vs. Newton vs. secant vs. fixed-point")
ax.legend(fontsize=8)
fig.tight_layout()

plt.show()
