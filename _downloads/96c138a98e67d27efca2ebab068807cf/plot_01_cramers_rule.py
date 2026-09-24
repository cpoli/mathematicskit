r"""
Cramer's rule vs. elimination
==================================

Cramer's rule gives every unknown as a ratio of two determinants,
``x_i = det(A_i) / det(A)``. It is exact in theory but needs ``n + 1``
determinants, each an ``O(n^3)`` factorization, so its cost grows like
``n^4`` against elimination's ``n^3``.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import cramer_solve, lu_solve_system

# %%
# A 3x3 system solved by determinants
# -----------------------------------------

A = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
b = np.array([8.0, -11.0, -3.0])
print("det(A) =", round(np.linalg.det(A), 10))
for i in range(3):
    A_i = A.copy()
    A_i[:, i] = b
    print(f"det(A_{i}) = {np.linalg.det(A_i):+.10f}")
print("Cramer solution:", np.round(cramer_solve(A, b), 10))

# %%
# Cost as n grows
# ---------------------

rng = np.random.default_rng(0)
sizes = [10, 20, 40, 80, 160]
t_cramer, t_lu = [], []
for n in sizes:
    M = rng.normal(size=(n, n))
    rhs = rng.normal(size=n)
    t0 = time.perf_counter()
    x_c = cramer_solve(M, rhs)
    t_cramer.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    x_lu = lu_solve_system(M, rhs)
    t_lu.append(time.perf_counter() - t0)
    print(f"n = {n:3d}: max |x_cramer - x_lu| = {np.max(np.abs(x_c - x_lu)):.1e}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(sizes, t_cramer, "o-", label="Cramer's rule")
ax.loglog(sizes, t_lu, "s-", label="LU solve")
ax.set_xlabel("n")
ax.set_ylabel("time (s)")
ax.legend()
fig.tight_layout()

plt.show()
