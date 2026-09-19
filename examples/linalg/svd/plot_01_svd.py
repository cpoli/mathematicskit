r"""
Singular value decomposition
=======================================

Computes the SVD ``A = U diag(S) V^T`` via :func:`numpy.linalg.svd`
(LAPACK's bidiagonalization-based ``?gesdd``), the numerically stable
production route -- rather than the textbook-definition route of
eigendecomposing ``A^T A``, which squares the condition number of the
problem being solved.
"""

# %%
import numpy as np

from mathkit.linalg import svd_decompose

# %%
# Reconstruct a random matrix from its SVD
# ----------------------------------------------

rng = np.random.default_rng(3)
A = rng.uniform(-3, 3, size=(5, 3))
result = svd_decompose(A)
print("singular values:", np.round(result.S, 6))
print("||U diag(S) Vt - A|| =", np.linalg.norm(result.U @ np.diag(result.S) @ result.Vt - A))
print("agrees with numpy.linalg.svd:", np.allclose(np.sort(result.S)[::-1], np.sort(np.linalg.svd(A, compute_uv=False))[::-1], atol=1e-6))
