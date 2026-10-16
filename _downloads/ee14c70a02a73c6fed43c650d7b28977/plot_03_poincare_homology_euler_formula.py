r"""
Poincaré's Analysis Situs: homology and the Euler-Poincaré formula (1895-1900)
==============================================================================

Poincaré made Betti's numbers a theory. Chains of simplices, their
boundaries, and the homology groups :math:`H_k = \ker\partial_k / \operatorname{im}\partial_{k+1}`,
together with torsion coefficients. One consequence generalizes Euler:
the alternating count of simplices equals the alternating sum of Betti
numbers,

.. math:: \sum_k (-1)^k f_k = \sum_k (-1)^k \beta_k ,

so :math:`\chi` depends only on the space, not on the triangulation. Below,
many triangulations of five spaces with wildly different simplex counts
land on the same :math:`\chi`.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.topology import circle, homology, klein_bottle, projective_plane, sphere, torus

families = {
    "sphere": [sphere(2), sphere(2, "cross_polytope")],
    "circle": [circle(n) for n in (3, 5, 9, 17)],
    "torus": [torus(n, m) for n, m in ((3, 3), (5, 4), (8, 6), (12, 9))],
    "Klein bottle": [klein_bottle(n, m) for n, m in ((3, 3), (6, 4), (10, 7))],
    "projective plane": [projective_plane()],
}

fig, ax = plt.subplots(figsize=(9, 5))
for i, (name, complexes) in enumerate(families.items()):
    for K in complexes:
        H = homology(K)
        assert H.euler_characteristic == K.euler_characteristic
        ax.scatter(len(K), K.euler_characteristic, s=50, color=f"C{i}", label=name if K is complexes[0] else None)
    print(f"{name:>16}: {H}  (chi = {H.euler_characteristic})")
ax.set_xscale("log")
ax.set_xlabel("number of simplices in the triangulation")
ax.set_ylabel(r"$\chi = \sum (-1)^k f_k = \sum (-1)^k \beta_k$")
ax.set_yticks([0, 1, 2])
ax.legend()
ax.set_title("The Euler characteristic is a topological invariant")
