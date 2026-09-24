r"""
Eckart-Young: the best low-rank approximation from the SVD
==========================================================

Truncating the SVD ``A = U diag(S) V^T`` to its ``k`` largest singular
values gives the best rank-``k`` approximation of ``A``: its error is
``sqrt(sigma_{k+1}^2 + ...)`` in the Frobenius norm and ``sigma_{k+1}``
in the spectral norm, and no other rank-``k`` matrix does better. The SVD
itself comes from :func:`numpy.linalg.svd` (LAPACK's stable
bidiagonalization route, not the eigendecomposition of ``A^T A``).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import svd_decompose

# %%
# A matrix with decaying singular values
# --------------------------------------
# A smooth "image" sampled on a 60x80 grid, plus a little noise.

rng = np.random.default_rng(3)
y, x = np.mgrid[-1:1:60j, -1:1:80j]
A = np.exp(-(x**2 + 2 * y**2) * 3) + 0.5 * np.cos(4 * x * y) + 0.02 * rng.normal(size=x.shape)
res = svd_decompose(A)
U, S, Vt = res.U, res.S, res.Vt
print("largest singular values:", np.round(S[:6], 4))
print("||U diag(S) Vt - A|| =", f"{np.linalg.norm(U @ np.diag(S) @ Vt - A):.1e}")


def truncate(k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k]


# %%
# Truncation error equals the discarded singular values
# -----------------------------------------------------

ks = np.arange(1, 21)
err_fro = np.array([np.linalg.norm(A - truncate(k)) for k in ks])
err_2 = np.array([np.linalg.norm(A - truncate(k), 2) for k in ks])
tail = np.array([np.sqrt(np.sum(S[k:] ** 2)) for k in ks])
print("Frobenius error == sqrt(sum of discarded sigma^2):", np.allclose(err_fro, tail))
print("spectral error  == sigma_{k+1}:                   ", np.allclose(err_2, S[ks]))

# %%
# No other rank-k matrix does better
# ----------------------------------
# Compare with rank-k approximations built by projecting ``A`` onto a
# random k-dimensional column space.

k = 5
competitors = []
for _ in range(200):
    Qk, _ = np.linalg.qr(rng.normal(size=(A.shape[0], k)))
    competitors.append(np.linalg.norm(A - Qk @ (Qk.T @ A)))
print(f"rank-{k}: SVD error {err_fro[k - 1]:.4f}, best of 200 random projections {min(competitors):.4f}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
axes[0].semilogy(ks, err_fro, "o-", label=r"$\|A - A_k\|_F$")
axes[0].semilogy(ks, S[ks], "s-", label=r"$\sigma_{k+1} = \|A - A_k\|_2$")
axes[0].set_xlabel("rank k")
axes[0].set_title("Truncation error")
axes[0].legend(fontsize=8)
for ax, (label, M) in zip(axes[1:], (("A", A), ("A_1", truncate(1)), (f"A_{k}", truncate(k)))):
    ax.imshow(M, cmap="viridis", vmin=A.min(), vmax=A.max())
    ax.set_title(f"${label}$")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()

plt.show()
