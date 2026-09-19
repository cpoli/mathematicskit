r"""
Nonlinear conjugate gradient vs. gradient descent
=======================================================

Compares Polak-Ribiere nonlinear CG against plain gradient descent on
the same ill-conditioned quadratic bowl -- CG's conjugate search
directions avoid the zig-zagging that slows gradient descent down.
"""

# %%
from mathematicskit.optimization import GradientDescent, NonlinearConjugateGradient, quadratic_bowl, quadratic_bowl_grad
from mathematicskit.optimization.utils.comparison import compare_optimizers
from mathematicskit.optimization.visualizers.plots import plot_convergence_comparison

# %%
# Compare both methods from the same starting point
# -----------------------------------------------------

optimizers = {
    "gradient_descent": GradientDescent(alpha=0.03, tol=1e-8, max_iter=2000),
    "conjugate_gradient": NonlinearConjugateGradient(tol=1e-8, max_iter=2000),
}
results = compare_optimizers(optimizers, quadratic_bowl, quadratic_bowl_grad, [5.0, -3.0])
for name, result in results.items():
    print(f"{name}: {result.iterations} iterations, x = {result.x}")

# %%
# Plot the convergence-rate comparison
# -----------------------------------------------------

plot_convergence_comparison(results, quadratic_bowl, f_star=0.0)
