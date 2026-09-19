r"""
Conjugate gradient and GMRES convergence
=============================================

Both are Krylov-subspace iterative solvers that never form a dense
factorization of ``A``. Conjugate gradient requires ``A`` symmetric
positive-definite; GMRES works for any nonsingular ``A``. This script
watches both converge and plots the residual-norm history.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import GMRES, ConjugateGradient, random_spd_matrix
from mathematicskit.linalg.visualizers.plots import plot_residual_history

# %%
# Conjugate gradient on an SPD system
# -----------------------------------------

A_spd = random_spd_matrix(20, seed=4)
b = np.ones(20)
result_cg = ConjugateGradient(tol=1e-10).solve(A_spd, b)
print(f"CG converged in {result_cg.iterations} iterations (n=20); final residual {result_cg.residual_history[-1]:.2e}")

# %%
# GMRES on a general nonsymmetric system
# ---------------------------------------------

rng = np.random.default_rng(5)
A_gen = rng.uniform(-1, 1, size=(20, 20)) + 20.0 * np.eye(20)
result_gmres = GMRES(tol=1e-10).solve(A_gen, b)
print(f"GMRES converged in {result_gmres.iterations} iterations (n=20); final residual {result_gmres.residual_history[-1]:.2e}")

fig, ax = plt.subplots(figsize=(6, 4))
plot_residual_history(result_cg, ax=ax)
plot_residual_history(result_gmres, ax=ax)
ax.legend()
fig.tight_layout()

plt.show()
