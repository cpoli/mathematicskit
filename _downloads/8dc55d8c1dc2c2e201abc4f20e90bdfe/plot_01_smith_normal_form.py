r"""
Smith's normal form of an integer matrix (1861)
===============================================

H. J. S. Smith showed that integer row and column operations bring any
integer matrix to a diagonal :math:`D = UAV` with
:math:`d_1 \mid d_2 \mid \cdots`. The invariant factors classify the
abelian group :math:`\mathbb{Z}^m / A\mathbb{Z}^n`: a factor 1 kills a
generator, :math:`d > 1` leaves a cyclic group :math:`\mathbb{Z}/d`, and
missing factors leave free :math:`\mathbb{Z}` summands. The same
reduction applied to a boundary matrix gives a space's torsion.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import smith_normal_form

A = np.array([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
result = smith_normal_form(A)
print("A =\n", A)
print("D = U A V =\n", result.D)
print("U A V == D:", np.array_equal(result.U @ A @ result.V, result.D))
print("det U, det V:", round(np.linalg.det(result.U)), round(np.linalg.det(result.V)))
print("invariant factors:", result.invariant_factors, " product:", np.prod(result.invariant_factors), " |det A|:", round(abs(np.linalg.det(A))))
print("Z^3 / A Z^3 =", " + ".join(f"Z/{d}" for d in result.invariant_factors if d > 1))

# %%
# The lattice picture in two dimensions
# -------------------------------------
#
# The columns of :math:`A = \begin{pmatrix} 4 & 2 \\ 2 & 4 \end{pmatrix}` span a sublattice of index
# :math:`|\det A| = 12`. Its Smith form :math:`\mathrm{diag}(2, 6)` says the
# quotient is :math:`\mathbb{Z}/2 \oplus \mathbb{Z}/6`, not :math:`\mathbb{Z}/12`.

B = np.array([[4, 2], [2, 4]])
smith = smith_normal_form(B)
print("\nSmith form of", B.tolist(), "is", smith.D.tolist())

grid = np.array([[i, j] for i in range(-8, 13) for j in range(-8, 13)])
sub = np.array([B @ [i, j] for i in range(-6, 7) for j in range(-6, 7)])
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (M, title) in zip(axes, ((B, "columns of A"), (np.linalg.inv(smith.U) @ smith.D, "after U: the diagonal basis")), strict=True):
    ax.scatter(*grid.T, s=6, color="0.7")
    lattice = np.array([M @ [i, j] for i in range(-8, 9) for j in range(-8, 9)])
    ax.scatter(*lattice.T, s=30, color="tab:blue")
    ax.quiver([0, 0], [0, 0], M[0], M[1], angles="xy", scale_units="xy", scale=1, color=["tab:red", "tab:orange"])
    ax.add_patch(plt.Polygon([[0, 0], M[:, 0], M[:, 0] + M[:, 1], M[:, 1]], alpha=0.2, color="tab:red"))
    ax.set_xlim(-8, 12)
    ax.set_ylim(-8, 12)
    ax.set_aspect("equal")
    ax.set_title(title)
fig.suptitle(f"A sublattice of index 12; invariant factors {smith.invariant_factors}")
