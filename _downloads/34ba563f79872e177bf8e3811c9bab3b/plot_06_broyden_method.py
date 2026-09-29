r"""
Broyden's quasi-Newton method
=============================

Newton's method for :math:`n` equations needs a fresh Jacobian at every
step: :math:`2n` extra evaluations of :math:`F` by central differences.
Charles Broyden's 1965 method computes the Jacobian once, then corrects
its approximation :math:`B_k` by the rank-one update

.. math::

   B_{k+1} = B_k + \frac{(\mathbf y_k - B_k \mathbf s_k)\,\mathbf s_k^T}{\mathbf s_k^T \mathbf s_k},

the smallest change that reproduces the latest step
(:math:`B_{k+1}\mathbf s_k = \mathbf y_k`). Each iteration then costs one
evaluation of :math:`F`. Convergence drops from quadratic to superlinear,
so Broyden takes more iterations but far fewer evaluations.

The test problem is Bratu's equation :math:`u'' + e^{u} = 0` on
:math:`[0, 1]` with :math:`u(0) = u(1) = 0`, discretized by central
differences at :math:`n = 40` interior points.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import Broyden, NewtonSystem

n = 40
h = 1.0 / (n + 1)


def bratu(u):
    padded = np.concatenate(([0.0], u, [0.0]))
    return (padded[:-2] - 2.0 * u + padded[2:]) / h**2 + np.exp(u)


u0 = np.zeros(n)
newton = NewtonSystem(bratu, u0, tol=1e-12).solve()
broyden = Broyden(bratu, u0, tol=1e-12).solve()
for result in (newton, broyden):
    n_eval, residual = result.function_evaluations, result.residual_norms[-1]
    print(f"{result.method:8s}: {result.iterations:2d} iterations, {n_eval:4d} evaluations of F, final residual {residual:.1e}")
print(f"max |u_newton - u_broyden| = {np.abs(newton.root - broyden.root).max():.1e}")

# %%
# Residual against iterations and against cost
# ---------------------------------------------


def evaluations_so_far(result, per_step, initial):
    return initial + per_step * np.arange(result.iterations + 1)


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))
ax0.plot(np.linspace(0, 1, n + 2), np.concatenate(([0.0], newton.root, [0.0])), "k")
ax0.set_xlabel("x")
ax0.set_ylabel("u(x)")
ax0.set_title("Solution of the discretized Bratu problem")

ax1.semilogy(newton.residual_norms, "o-", label="Newton")
ax1.semilogy(broyden.residual_norms, "s-", label="Broyden")
ax1.set_xlabel("iteration k")
ax1.set_ylabel(r"$\|F(\mathbf{u}_k)\|_2$")
ax1.set_title("Newton needs fewer iterations...")
ax1.legend()

ax2.semilogy(evaluations_so_far(newton, 2 * n + 1, 1), newton.residual_norms, "o-", label="Newton (2n + 1 per step)")
ax2.semilogy(evaluations_so_far(broyden, 1, 1 + 2 * n), broyden.residual_norms, "s-", label="Broyden (1 per step)")
ax2.set_xlabel("evaluations of F")
ax2.set_title("...but Broyden needs fewer evaluations")
ax2.legend()
fig.tight_layout()

plt.show()
