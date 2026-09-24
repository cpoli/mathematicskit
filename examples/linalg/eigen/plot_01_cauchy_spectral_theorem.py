r"""
Cauchy's spectral theorem: principal axes of a quadratic form
=============================================================

Cauchy's 1829 memoir showed that a real symmetric matrix has only real
eigenvalues and an orthonormal basis of eigenvectors, so
``A = V diag(lambda) V^T``. Geometrically, the eigenvectors are the
principal axes of the quadric ``x^T A x = 1``. A non-symmetric matrix
enjoys neither property.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import eigen_general, eigen_symmetric

# %%
# Principal axes of an ellipse
# ----------------------------
# The quadratic form ``5 x^2 + 4 x y + 2 y^2 = 1`` is an ellipse whose axes
# point along the eigenvectors of its symmetric matrix, with semi-axis
# lengths ``1 / sqrt(lambda)``.

A = np.array([[5.0, 2.0], [2.0, 2.0]])
eig = eigen_symmetric(A)
lams, V = eig.eigenvalues, eig.eigenvectors
print("eigenvalues:", lams)
print("V^T V = I:", np.allclose(V.T @ V, np.eye(2)))

theta = np.linspace(0.0, 2.0 * np.pi, 400)
circle = np.vstack([np.cos(theta), np.sin(theta)])
ellipse = V @ np.diag(1.0 / np.sqrt(lams)) @ circle  # x = V Lambda^{-1/2} u, |u| = 1

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(ellipse[0], ellipse[1], label=r"$x^T A x = 1$")
for k in range(2):
    axis = V[:, k] / np.sqrt(lams[k])
    ax.plot([-axis[0], axis[0]], [-axis[1], axis[1]], lw=2, label=rf"axis along $v_{k + 1}$, $\lambda_{k + 1} = {lams[k]:.0f}$")
ax.set_aspect("equal")
ax.set_title("Eigenvectors of a symmetric matrix = principal axes")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()

# %%
# A larger symmetric matrix: real spectrum, orthonormal eigenvectors
# ------------------------------------------------------------------

rng = np.random.default_rng(2)
M = rng.normal(size=(6, 6))
S = M + M.T
eig_s = eigen_symmetric(S)
V = eig_s.eigenvectors
print("eigenvalues (all real):", np.round(eig_s.eigenvalues, 6))
print(f"||V^T V - I|| = {np.linalg.norm(V.T @ V - np.eye(6)):.1e}")
print(f"||V diag(lambda) V^T - S|| = {np.linalg.norm(V @ np.diag(eig_s.eigenvalues) @ V.T - S):.1e}")

# %%
# Without symmetry the theorem fails
# ----------------------------------
# The non-symmetric part ``M`` alone has complex eigenvalues and
# eigenvectors that are not orthogonal.

eig_m = eigen_general(M)
print("eigenvalues of M:", np.round(eig_m.eigenvalues, 4))
W = eig_m.eigenvectors
print(f"||W^H W - I|| = {np.linalg.norm(W.conj().T @ W - np.eye(6)):.2f}")

plt.show()
