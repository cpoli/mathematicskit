r"""
Nelder-Mead: minimizing without derivatives
=================================================

The Nelder-Mead simplex method uses function values only. On the
Rosenbrock function it needs more function evaluations than BFGS, which
uses gradients, but it also works on objectives that have no gradient at
all.
"""

# %%
from mathematicskit.optimization import BFGS, NelderMead, rosenbrock, rosenbrock_grad
from mathematicskit.optimization.visualizers.plots import plot_contour_path

# %%
# Nelder-Mead vs. BFGS on the Rosenbrock function
# -----------------------------------------------------

x0 = [-1.2, 1.0]
nm = NelderMead(tol=1e-8, max_iter=5000).minimize(rosenbrock, None, x0)
bfgs = BFGS(tol=1e-8).minimize(rosenbrock, rosenbrock_grad, x0)
print(f"Nelder-Mead: x = {nm.x.round(6)}, {nm.iterations} iterations, {nm.extra['nfev']} function evaluations")
print(f"BFGS:        x = {bfgs.x.round(6)}, {bfgs.iterations} iterations")

ax = plot_contour_path(rosenbrock, nm, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0), label="Nelder-Mead (1965)")
plot_contour_path(rosenbrock, bfgs, ax=ax, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0), label="BFGS")

# %%
# A non-smooth objective
# -----------------------------------------------------

result = NelderMead(tol=1e-10).minimize(lambda x: abs(x[0] - 1.0) + 2.0 * abs(x[1] + 0.5), None, [0.0, 0.0])
print(f"minimizer of |x - 1| + 2|y + 0.5|: {result.x.round(6)}")
