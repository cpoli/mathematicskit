r"""
Frobenius's character tables
============================

In 1896 Frobenius attached to each finite group a square table of
numbers: one row per irreducible character
:math:`\chi(g) = \operatorname{tr}\rho(g)`, one column per conjugacy
class. The rows are orthonormal, the squares of the first column sum to
:math:`|G|`, and the table encodes much of the group's structure in a
handful of numbers. Here the tables are computed by Burnside's
algorithm, from the class multiplication constants alone, without
constructing any representation.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.abstract_algebra import DihedralGroup, PermutationGroup, QuaternionGroup, character_table


def show(name, result):
    print(f"{name}: class sizes {result.class_sizes}, degrees {result.degrees}")
    for row in result.table:
        print("   " + "  ".join(f"{z.real:6.3f}" if abs(z.imag) < 1e-9 else f"{z.real:+.2f}{z.imag:+.2f}i" for z in row))


# %%
# S_4, the rotations of a cube
# ------------------------------

s4 = character_table(PermutationGroup(4))
show("S_4", s4)
sizes = np.array(s4.class_sizes)
gram = (s4.table * sizes) @ s4.table.conj().T / 24
print(f"row orthogonality: max |<chi_i, chi_j> - delta_ij| = {np.abs(gram - np.eye(5)).max():.1e}")
print(f"sum of squared degrees = {sum(d**2 for d in s4.degrees)} = |S_4|")

# %%
# Two groups, one table
# -----------------------
#
# The dihedral group :math:`D_4` (symmetries of a square) and the
# quaternion group :math:`Q_8` are not isomorphic: :math:`D_4` has five
# elements of order 2, :math:`Q_8` only one. Yet their character tables
# are identical, a classic warning that the table does not determine the
# group.

d4, q8 = character_table(DihedralGroup(4)), character_table(QuaternionGroup())
show("D_4", d4)
show("Q_8", q8)
print(f"identical tables: {np.allclose(d4.table, q8.table)}")
for name, group in (("D_4", DihedralGroup(4)), ("Q_8", QuaternionGroup())):
    print(f"elements of order 2 in {name}: {sum(group.element_order(g) == 2 for g in group.elements)}")

# %%
# A_5 and the golden ratio
# --------------------------
#
# The rotation group of the icosahedron is :math:`A_5`. Its two
# three-dimensional characters take the values
# :math:`(1 \pm \sqrt5)/2` on the two classes of 5-cycles, the golden
# ratio of the icosahedron's geometry.

a5_group = PermutationGroup(5, generators=[(1, 2, 0, 3, 4), (0, 1, 3, 4, 2)])
a5 = character_table(a5_group)
show("A_5", a5)

# %%
# The tables as images
# ----------------------

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (name, result) in zip(axes, [("$S_4$", s4), ("$D_4$", d4), ("$Q_8$", q8), ("$A_5$", a5)], strict=True):
    values = result.table.real
    image = ax.imshow(values, cmap="RdBu_r", vmin=-5, vmax=5)
    for (i, j), v in np.ndenumerate(values):
        ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(result.classes)), [f"|C|={c}" for c in result.class_sizes], fontsize=8)
    ax.set_yticks(range(len(result.classes)), [rf"$\chi_{i + 1}$" for i in range(len(result.classes))])
    ax.set_title(f"{name}, order {sum(result.class_sizes)}")
fig.colorbar(image, ax=axes, shrink=0.8)

plt.show()
