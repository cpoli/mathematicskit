r"""
The classification of surfaces (Dehn and Heegaard 1907, Brahana 1921)
=====================================================================

Every compact connected surface is a sphere with handles or with
cross-caps, minus some discs. Three numbers decide which: orientability,
the Euler characteristic, and the number of boundary circles. Here
:func:`~mathematicskit.topology.classify_surface` names nine
triangulated surfaces from those invariants alone.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.topology import (
    SimplicialComplex,
    barycentric_subdivision,
    classify_surface,
    klein_bottle,
    mobius_strip,
    projective_plane,
    simplex,
    sphere,
    torus,
)

T = torus(6, 4)
surfaces = {
    "boundary of a tetrahedron": sphere(2),
    "triangle": simplex(2),
    "ring of squares": T.induced_subcomplex([v for v in T.vertices if v // 4 in range(4)]),
    "doughnut grid": T,
    "torus minus a triangle": SimplicialComplex(T.simplices(2)[1:]),
    "twisted band": mobius_strip(),
    "6-vertex RP^2": projective_plane(),
    "subdivided RP^2": barycentric_subdivision(projective_plane()),
    "twisted grid": klein_bottle(),
}

rows = []
for label, K in surfaces.items():
    c = classify_surface(K)
    rows.append([label, f"{K.f_vector}", str(c.euler_characteristic), "yes" if c.orientable else "no", str(c.boundary_components), c.name])
    print(f"{label:>26} -> {c.name}")

# %%
# The invariants side by side
# ---------------------------

fig, ax = plt.subplots(figsize=(12, 3.6))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=["triangulation", "(V, E, F)", "χ", "orientable", "boundary", "surface"], loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(range(6))
table.scale(1, 1.4)
ax.set_title("Orientability, χ and boundary circles determine the surface")
