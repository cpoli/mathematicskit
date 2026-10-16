r"""
Poincaré's fundamental group (1895)
===================================

Loops from a base point, up to deformation, form a group,
:math:`\pi_1`. For a complex it can be read off combinatorially:
contract a spanning tree of the edges, keep one generator per remaining
edge, and impose one relation per triangle. Tietze moves then shrink the
presentation. The torus gives :math:`\langle a, b \mid aba^{-1}b^{-1}\rangle`
(up to naming), the projective plane :math:`\langle a \mid a^2\rangle`, and
abelianizing gives back :math:`H_1`.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.topology import (
    SimplicialComplex,
    abelianization,
    circle,
    fundamental_group,
    homology,
    klein_bottle,
    mobius_strip,
    projective_plane,
    sphere,
    torus,
)

wedge = SimplicialComplex([[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]])
spaces = {
    "circle": circle(5),
    "figure eight": wedge,
    "sphere": sphere(2),
    "torus": torus(3, 3),
    "Möbius strip": mobius_strip(5),
    "projective plane": projective_plane(),
    "Klein bottle": klein_bottle(3, 3),
}
for name, K in spaces.items():
    raw = fundamental_group(K, simplify=False)
    P = fundamental_group(K)
    print(f"{name:>16}: {len(raw.generators):2d} edge generators, {len(raw.relations):2d} relations -> {P}")
    print(f"{'':>16}  abelianized: {abelianization(P)}, H_1 = {homology(K).group(1)}")

# %%
# The spanning tree on the 3 x 3 torus
# ------------------------------------
#
# Drawn on the square with opposite sides glued: the 8 tree edges are
# contracted, and the other 19 edges are generators. The 18 triangles
# then cut them down to two, :math:`a` and :math:`b`, with one relation.

T = torus(3, 3)
pos = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}
P_raw = fundamental_group(T, simplify=False)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for i in range(3):
    for j in range(3):
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            a, b = i * 3 + j, ((i + di) % 3) * 3 + (j + dj) % 3
            edge = (min(a, b), max(a, b))
            is_generator = edge in P_raw.generators
            ax.plot([i, i + di], [j, j + dj], color="0.7" if is_generator else "tab:green", lw=1 if is_generator else 3)
for v, (i, j) in pos.items():
    ax.annotate(str(v), (i, j), ha="center", va="center", bbox={"boxstyle": "circle", "fc": "w"})
ax.add_patch(plt.Rectangle((0, 0), 3, 3, fill=False, ls="--"))
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"spanning tree (green) of the torus; π₁ = {fundamental_group(T)}")
