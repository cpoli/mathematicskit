r"""
The QR algorithm and the Schur form
========================================

Factor ``A_k = Q_k R_k`` and multiply back in reverse order,
``A_{k+1} = R_k Q_k = Q_k^T A_k Q_k``. Every iterate is similar to
``A``, and the subdiagonal entries decay like
``|lambda_{i+1} / lambda_i|^k``, so the iterates converge to an upper
triangular Schur form with the eigenvalues on the diagonal. Production
codes add Hessenberg reduction and Francis's shifts; the result is what
:func:`scipy.linalg.schur` returns.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import hessenberg_reduce, householder_qr, schur_decompose

# %%
# Unshifted QR iteration
# ----------------------------

rng = np.random.default_rng(7)
lams = np.array([9.0, 6.0, 4.0, 2.5, 1.0])
S = rng.normal(size=(5, 5)) + 3.0 * np.eye(5)
A = S @ np.diag(lams) @ np.linalg.inv(S)  # nonsymmetric, eigenvalues lams
H, _ = hessenberg_reduce(A)

Ak = H.copy()
subdiag = []
for _ in range(60):
    qr = householder_qr(Ak)
    Ak = qr.R @ qr.Q
    subdiag.append(np.abs(np.diag(Ak, -1)))
subdiag = np.array(subdiag)
print("diag(A_60):   ", np.round(np.diag(Ak), 8))

# %%
# Compare with LAPACK's Schur decomposition
# -----------------------------------------------

schur = schur_decompose(A)
print("diag(T) Schur:", np.round(np.sort(np.diag(schur.T))[::-1], 8))
print("T below the diagonal is zero:", np.allclose(np.tril(schur.T, -1), 0.0))
print(f"||Z T Z^T - A|| = {np.linalg.norm(schur.Z @ schur.T @ schur.Z.T - A):.1e}")

fig, ax = plt.subplots(figsize=(6, 4))
k = np.arange(1, len(subdiag) + 1)
for i in range(4):
    ax.semilogy(k, subdiag[:, i], label=f"|a_{i + 2},{i + 1}|")
    ax.semilogy(k, subdiag[0, i] * (lams[i + 1] / lams[i]) ** (k - 1), "k:", lw=0.8)
ax.set_xlabel("QR step k")
ax.set_ylabel("subdiagonal magnitude")
ax.set_title("Subdiagonal decay ~ (lambda_{i+1}/lambda_i)^k")
ax.legend()
fig.tight_layout()

plt.show()
