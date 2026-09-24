r"""
Tanh-sinh quadrature: integrating endpoint singularities
==============================================================

Integrates 1/sqrt(x) and log(x) over [0, 1], whose integrands blow up
at 0. The tanh-sinh substitution crowds nodes toward the endpoints with
double-exponentially small weights, and converges far faster than
Gauss-Legendre quadrature on these integrands.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.calculus import GaussianQuadrature, TanhSinhQuadrature

# %%
# Where the nodes go
# -----------------------------------------------------

rule = TanhSinhQuadrature(h=0.5, t_max=3.0)
t = rule.h * np.arange(-6, 7)
nodes = 0.5 + 0.5 * np.tanh(0.5 * np.pi * np.sinh(t))
print("tanh-sinh nodes on [0, 1] (h = 0.5):")
print("  " + ", ".join(f"{x:.3g}" for x in nodes))

# %%
# Convergence on singular integrands
# -----------------------------------------------------

cases = {r"$1/\sqrt{x}$": (lambda x: 1 / np.sqrt(x), 2.0), r"$\log x$": (np.log, -1.0)}
hs = [1.0, 0.5, 0.25, 0.125, 0.0625]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, (f, exact)) in zip(axes, cases.items()):
    evals, errors = [], []
    for h in hs:
        result = TanhSinhQuadrature(h=h).integrate(f, 0.0, 1.0)
        evals.append(result.n_evaluations)
        errors.append(max(abs(result.value - exact), 1e-17))
    gauss_n = [4, 8, 16, 32, 64, 128]
    gauss = [abs(GaussianQuadrature(n).integrate(f, 0.0, 1.0).value - exact) for n in gauss_n]
    ax.loglog(evals, errors, "o-", label="tanh-sinh")
    ax.loglog(gauss_n, gauss, "s-", label="Gauss-Legendre")
    ax.set_xlabel("function evaluations")
    ax.set_title(name)
    ax.legend()
axes[0].set_ylabel("|error|")
