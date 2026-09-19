r"""
Householder vs. Gram-Schmidt QR
====================================

Both factor ``A = Q R``, but Householder reflections stay orthogonal to
machine precision even for ill-conditioned ``A``, while classical
Gram-Schmidt can lose orthogonality badly -- modified Gram-Schmidt is a
partial (but not complete) remedy. This is the textbook motivation for
preferring Householder in production numerical software.
"""

# %%
import numpy as np

from mathkit.linalg import gram_schmidt_qr, householder_qr
from mathkit.linalg.systems.qr import orthogonality_error

# %%
# A well-conditioned matrix: all three methods agree
# --------------------------------------------------------

rng = np.random.default_rng(0)
A = rng.uniform(-2, 2, size=(6, 4))
for label, result in (
    ("householder", householder_qr(A)),
    ("gram_schmidt (modified)", gram_schmidt_qr(A, modified=True)),
    ("gram_schmidt (classical)", gram_schmidt_qr(A, modified=False)),
):
    print(f"{label:>28s}: ||QR - A|| = {np.linalg.norm(result.Q @ result.R - A):.2e}, orthogonality error = {orthogonality_error(result.Q):.2e}")

# %%
# An ill-conditioned matrix: orthogonality breaks down differently
# ---------------------------------------------------------------------
# Nearly-collinear columns (Trefethen & Bau's classic example) expose the
# gap between the three methods.

eps = 1e-8
A_ill = np.array([[1.0, 1.0, 1.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]])
for label, result in (
    ("householder", householder_qr(A_ill)),
    ("gram_schmidt (modified)", gram_schmidt_qr(A_ill, modified=True)),
    ("gram_schmidt (classical)", gram_schmidt_qr(A_ill, modified=False)),
):
    print(f"{label:>28s}: orthogonality error = {orthogonality_error(result.Q):.3e}")
