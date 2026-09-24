r"""
Nesterov acceleration: O(1/k^2) versus O(1/k)
===================================================

On an ill-conditioned convex quadratic, plain gradient descent reduces
the objective gap like :math:`O(1/k)` until the strongly convex regime
kicks in; Nesterov's accelerated gradient, with the same step size,
achieves the optimal first-order rate :math:`O(1/k^2)`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import GradientDescent, NesterovAcceleratedGradient

# %%
# f(x) = (x^2 + 0.001 y^2) / 2, step 1/L = 1
# -----------------------------------------------------

d = np.array([1.0, 1e-3])
f = lambda x: 0.5 * float(np.sum(d * x * x))
grad = lambda x: d * x
x0 = np.array([1.0, 1.0])

gd = GradientDescent(alpha=1.0, tol=0.0, max_iter=2000).minimize(f, grad, x0)
nag = NesterovAcceleratedGradient(alpha=1.0, tol=0.0, max_iter=2000).minimize(f, grad, x0)
print(f"after 2000 iterations: gradient descent f = {gd.fun:.3e}, Nesterov f = {nag.fun:.3e}")

k = np.arange(1, 2001)
fig, ax = plt.subplots()
ax.loglog(k, [f(x) for x in gd.path[1:]], label="gradient descent")
ax.loglog(k, [f(x) for x in nag.path[1:]], label="Nesterov (1983)")
ax.loglog(k, 2.0 * np.sum(x0**2) / (k + 1) ** 2, "k--", label=r"bound $2L\|x_0-x^*\|^2/(k+1)^2$")
ax.set_xlabel("iteration k")
ax.set_ylabel(r"$f(x_k) - f^*$")
ax.legend()
ax.set_title("Accelerated gradient descent")
