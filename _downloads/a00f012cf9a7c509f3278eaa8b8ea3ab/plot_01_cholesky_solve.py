r"""
Cholesky decomposition for SPD systems
==========================================

Cholesky factors a symmetric positive-definite ``A = L L^T`` at roughly
half the cost of LU, and the factorization succeeding at all (every
``sqrt`` argument staying positive) is itself a certificate of
positive-definiteness.
"""

# %%
import numpy as np

from mathkit.linalg import cholesky_decompose, cholesky_solve, is_symmetric_positive_definite, random_spd_matrix

# %%
# Factor and solve
# --------------------

A = random_spd_matrix(5, seed=1)
result = cholesky_decompose(A)
print("L @ L.T == A:", np.allclose(result.L @ result.L.T, A))

x_true = np.arange(1.0, 6.0)
b = A @ x_true
x = cholesky_solve(result, b)
print("recovered x:", np.round(x, 8))
print("matches x_true:", np.allclose(x, x_true, atol=1e-8))

# %%
# Positive-definiteness as a byproduct of the factorization succeeding
# --------------------------------------------------------------------

print("is_symmetric_positive_definite(A):", is_symmetric_positive_definite(A))
print("is_symmetric_positive_definite(non-SPD):", is_symmetric_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]])))
