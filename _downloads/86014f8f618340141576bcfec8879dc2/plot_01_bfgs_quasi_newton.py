r"""
BFGS: Newton-like convergence without second derivatives
==============================================================

On the Rosenbrock function from ``(-1.2, 1.0)``, BFGS builds an
inverse-Hessian approximation from gradients alone. It needs only a few
dozen iterations, as few as Newton's method with the exact Hessian at
every step, while gradient descent needs thousands. Near the minimum its
convergence is superlinear: the ratio of successive errors tends to zero.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import BFGS, GradientDescentLineSearch, NewtonMethod, rosenbrock, rosenbrock_grad, rosenbrock_hess
from mathematicskit.optimization.visualizers.plots import plot_contour_path, plot_convergence_comparison

# %%
# BFGS against Newton's method and gradient descent
# -------------------------------------------------

x0 = [-1.2, 1.0]
results = {
    "gradient descent": GradientDescentLineSearch(tol=1e-6, max_iter=5000).minimize(rosenbrock, rosenbrock_grad, x0),
    "Newton (exact Hessian)": NewtonMethod(tol=1e-8).minimize(rosenbrock, rosenbrock_grad, x0, hess=rosenbrock_hess),
    "BFGS (gradients only)": BFGS(tol=1e-8).minimize(rosenbrock, rosenbrock_grad, x0),
}
for name, result in results.items():
    print(f"{name}: {result.iterations} iterations, converged = {result.converged}, x = {result.x.round(6)}")

# %%
# Superlinear convergence of BFGS
# -------------------------------
# The error ratio :math:`\|x_{k+1}-x^*\| / \|x_k-x^*\|` stays bounded away
# from zero for a linearly convergent method; for BFGS it tends to zero.

bfgs = results["BFGS (gradients only)"]
err = np.linalg.norm(bfgs.path - np.array([1.0, 1.0]), axis=1)
ratios = err[1:] / err[:-1]
print("last BFGS error ratios:", np.round(ratios[-6:-1], 3))

# %%
# Convergence rates and the BFGS path
# -----------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
plot_convergence_comparison(results, rosenbrock, f_star=0.0, ax=ax1)
ax1.set_xlim(0, 100)
ax1.set_title("BFGS keeps pace with Newton's method")
plot_contour_path(rosenbrock, bfgs, ax=ax2, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0), label="BFGS")
ax2.set_title("BFGS iterates on the Rosenbrock valley")
fig.tight_layout()
