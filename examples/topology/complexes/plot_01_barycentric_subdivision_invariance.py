r"""
Brouwer, Alexander, and the invariance of homology under subdivision (1911-1915)
================================================================================

Is homology a property of the *space* or of the chosen triangulation?
Brouwer's simplicial approximation (1911) answers it. Any continuous
map is close to a simplicial one after subdividing finely enough.
Alexander (1915) used this to prove that homology is a topological
invariant. Barycentric subdivision replaces every simplex by the chains
of its faces. The simplex counts grow about sixfold per round for
surfaces, while the Betti numbers do not change.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.topology import barycentric_subdivision, betti_numbers, projective_plane, simplex, torus
from mathematicskit.topology.visualizers import plot_complex

for name, K in (("torus", torus(3, 3)), ("projective plane", projective_plane())):
    for level in range(3):
        print(f"{name:>16}, sd^{level}: f = {K.f_vector!s:>18}  chi = {K.euler_characteristic}  Betti = {betti_numbers(K)}  mod 2: {betti_numbers(K, field=2)}")
        if level < 2:
            K = barycentric_subdivision(K)

# %%
# Subdividing a triangle
# ----------------------

triangle = simplex(2)
triangle.coordinates = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
K = triangle
for level, ax in enumerate(axes):
    plot_complex(K, ax=ax, face_values=list(range(len(K.simplices(2)))), cmap="tab20", alpha=0.6, vertex_size=6)
    ax.set_title(f"sd^{level}: f = {K.f_vector}, χ = {K.euler_characteristic}")
    ax.axis("off")
    K = barycentric_subdivision(K)
fig.suptitle("Barycentric subdivision changes the triangulation, not the space")
