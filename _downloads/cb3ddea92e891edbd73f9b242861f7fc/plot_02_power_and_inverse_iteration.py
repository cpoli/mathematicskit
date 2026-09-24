r"""
Von Mises power iteration and Wielandt inverse iteration
========================================================

Power iteration ``v_{k+1} = A v_k / ||A v_k||`` converges to the dominant
eigenvector at the rate ``|lambda_2 / lambda_1|`` per step (squared for
the Rayleigh quotient of a symmetric matrix). Inverse iteration applies
the same step to ``(A - mu I)^{-1}`` and homes in on the eigenvalue
nearest the shift ``mu``, at the much faster rate
``|lambda - mu| / |lambda' - mu|``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import inverse_iteration, power_iteration

# %%
# A symmetric matrix with a known spectrum
# ----------------------------------------

lams = np.array([10.0, 8.0, 3.0, 1.0, -2.0])
rng = np.random.default_rng(5)
Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
A = Q @ np.diag(lams) @ Q.T

result = power_iteration(A)
print(f"power iteration: lambda ~ {result.eigenvalues[0]:.10f} after {result.iterations} iterations")
mu = 2.8
result_inv = inverse_iteration(A, mu=mu)
print(f"inverse iteration (mu = {mu}): lambda ~ {result_inv.eigenvalues[0]:.10f} after {result_inv.iterations} iterations")

# %%
# Error of the Rayleigh quotient, iteration by iteration
# ------------------------------------------------------
# Rerunning with a growing iteration cap (the default start vector is
# fixed) records the whole convergence history.

ks = np.arange(1, 61)
err_power = [abs(power_iteration(A, tol=0.0, max_iter=int(k)).eigenvalues[0] - 10.0) for k in ks]
ks_inv = np.arange(1, 9)
err_inv = [abs(inverse_iteration(A, mu=mu, tol=0.0, max_iter=int(k)).eigenvalues[0] - 3.0) for k in ks_inv]

rate_power = (8.0 / 10.0) ** 2
rate_inv = (abs(3.0 - mu) / abs(1.0 - mu)) ** 2
print(f"predicted error ratios per step: power {rate_power:.3f}, inverse {rate_inv:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(ks, np.maximum(err_power, 1e-16), "o-", ms=3, label=r"power iteration $\to \lambda_1 = 10$")
ax.semilogy(ks, err_power[0] * rate_power ** (ks - 1), "k--", lw=1, label=r"$(\lambda_2/\lambda_1)^{2k}$")
ax.semilogy(ks_inv, np.maximum(err_inv, 1e-16), "s-", ms=4, label=rf"inverse iteration, $\mu = {mu}$ $\to \lambda = 3$")
ax.set_xlabel("iteration k")
ax.set_ylabel("|Rayleigh quotient - eigenvalue|")
ax.set_title("Power vs. inverse iteration")
ax.legend(fontsize=8)
fig.tight_layout()

plt.show()
