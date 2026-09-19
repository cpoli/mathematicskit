r"""
Fixed step vs. backtracking line search on an ill-conditioned bowl
=========================================================================

Gradient descent zig-zags across a narrow, elongated quadratic bowl.
Backtracking line search adapts its step length automatically and
converges in far fewer iterations than any single fixed step size.
"""

# %%
from mathematicskit.optimization import GradientDescent, GradientDescentLineSearch, quadratic_bowl, quadratic_bowl_grad
from mathematicskit.optimization.visualizers.plots import plot_contour_path

# %%
# Run both methods from the same starting point
# -----------------------------------------------------

x0 = [1.8, 2.5]
result_fixed = GradientDescent(alpha=0.09, tol=1e-8, max_iter=5000).minimize(quadratic_bowl, quadratic_bowl_grad, x0)
result_ls = GradientDescentLineSearch(tol=1e-8, max_iter=5000).minimize(quadratic_bowl, quadratic_bowl_grad, x0)

print(f"fixed step:   {result_fixed.iterations} iterations, x = {result_fixed.x}")
print(f"line search:  {result_ls.iterations} iterations, x = {result_ls.x}")

# %%
# Plot both iterate paths over the bowl's contours
# -----------------------------------------------------

plot_contour_path(quadratic_bowl, result_fixed, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0))
plot_contour_path(quadratic_bowl, result_ls, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0))
