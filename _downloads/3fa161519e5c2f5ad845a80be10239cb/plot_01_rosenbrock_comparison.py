r"""
All five unconstrained methods on the Rosenbrock function
================================================================

Gradient descent, nonlinear conjugate gradient, Newton's method, and
BFGS all started from the classic ``(-1.2, 1.0)`` starting point,
compared by convergence rate. Newton's method (using the exact
Hessian) and BFGS reach the minimum in far fewer iterations than
gradient descent's slow crawl along the curved valley floor.
"""

# %%
from mathematicskit.optimization import BFGS, GradientDescentLineSearch, NewtonMethod, NonlinearConjugateGradient, rosenbrock, rosenbrock_grad, rosenbrock_hess
from mathematicskit.optimization.visualizers.plots import plot_contour_path, plot_convergence_comparison

# %%
# Run every method from the same starting point
# -----------------------------------------------------

x0 = [-1.2, 1.0]
gd = GradientDescentLineSearch(tol=1e-6, max_iter=5000).minimize(rosenbrock, rosenbrock_grad, x0)
cg = NonlinearConjugateGradient(tol=1e-6, max_iter=5000).minimize(rosenbrock, rosenbrock_grad, x0)
newton = NewtonMethod(tol=1e-8).minimize(rosenbrock, rosenbrock_grad, x0, hess=rosenbrock_hess)
bfgs = BFGS(tol=1e-8).minimize(rosenbrock, rosenbrock_grad, x0)

results = {"gradient_descent": gd, "conjugate_gradient": cg, "newton": newton, "bfgs": bfgs}
for name, result in results.items():
    print(f"{name}: {result.iterations} iterations, x = {result.x}")

# %%
# Convergence-rate comparison
# -----------------------------------------------------

plot_convergence_comparison(results, rosenbrock, f_star=0.0)

# %%
# BFGS's path over the Rosenbrock valley
# -----------------------------------------------------

plot_contour_path(rosenbrock, bfgs, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0))
