r"""
Robbins-Monro: stochastic gradient descent with decaying steps
====================================================================

Minimizes a quadratic using only noisy gradient observations. With
Robbins and Monro's step sizes :math:`a_k = a_0/(k+1)`, the noise
averages out and the error shrinks like :math:`1/\sqrt{k}`. A constant
step size instead stalls at a noise floor.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import robbins_monro

# %%
# Noisy gradients of f(x) = ||x - mu||^2 / 2
# -----------------------------------------------------

mu = np.array([2.0, -1.0])


def noisy_grad(x, rng):
    return x - mu + rng.normal(scale=2.0, size=2)


n = 20000
result = robbins_monro(noisy_grad, [0.0, 0.0], a0=1.0, n_iter=n, seed=0)
print(f"estimate after {n} steps: {result.x.round(4)} (true minimizer {mu})")

# Constant step size, for comparison.
rng = np.random.default_rng(0)
x = np.zeros(2)
const_err = []
for _ in range(n):
    x = x - 0.05 * noisy_grad(x, rng)
    const_err.append(np.linalg.norm(x - mu))

# %%
# Error versus iteration
# -----------------------------------------------------

k = np.arange(1, n + 1)
fig, ax = plt.subplots()
ax.loglog(k, np.linalg.norm(result.path[1:] - mu, axis=1), label=r"Robbins-Monro, $a_k = 1/(k+1)$")
ax.loglog(k, const_err, alpha=0.7, label=r"constant step $a = 0.05$")
ax.loglog(k, 2.0 * np.sqrt(2.0 / k), "k--", label=r"$\propto 1/\sqrt{k}$")
ax.set_xlabel("iteration k")
ax.set_ylabel(r"$\|x_k - \mu\|$")
ax.legend()
ax.set_title("Stochastic approximation (Robbins-Monro, 1951)")
