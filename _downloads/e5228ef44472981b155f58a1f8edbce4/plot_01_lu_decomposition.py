r"""
LU decomposition with partial pivoting
==========================================

Factors ``P A = L U`` and reuses the factorization to solve multiple
right-hand sides and compute the determinant, both far cheaper than
refactoring from scratch each time.
"""

# %%
import numpy as np

from mathkit.linalg import lu_decompose, lu_det, lu_solve

# %%
# Factor once, solve for several right-hand sides
# ----------------------------------------------------

A = np.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]])
result = lu_decompose(A)
print("P @ A == L @ U:", np.allclose(result.P @ A, result.L @ result.U))

for b in (np.array([5.0, 12.0, 24.0]), np.array([1.0, 0.0, -1.0])):
    x = lu_solve(result, b)
    print(f"b={b} -> x={np.round(x, 6)}, residual={np.linalg.norm(A @ x - b):.2e}")

# %%
# Determinant via the pivoted triangular factors
# ----------------------------------------------------

det_lu = lu_det(A)
det_numpy = np.linalg.det(A)
print(f"det via LU: {det_lu:.10f}, numpy.linalg.det: {det_numpy:.10f}")
