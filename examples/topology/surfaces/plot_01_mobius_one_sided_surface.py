r"""
Möbius and Listing: the one-sided surface (1858-1865)
=====================================================

Glue the ends of a strip with a half twist and the surface has only one
side. In triangle terms, its triangles cannot all be oriented so that
neighbours traverse their shared edge in opposite directions. Propagating
orientations around the strip comes back reversed. The untwisted band
(an annulus) has no such obstruction. Both have :math:`\chi = 0`, so
orientability is what tells them apart, along with the number of
boundary circles: one for the Möbius strip, two for the annulus.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import SimplicialComplex, boundary_components, is_orientable, mobius_strip, torus
from mathematicskit.topology.visualizers import plot_complex

M = mobius_strip(n=24, m=3, width=0.8)
T = torus(24, 4)
annulus = T.induced_subcomplex([v for v in T.vertices if v % 4 in (0, 1, 2)])
annulus = SimplicialComplex(annulus.simplices(), T.coordinates)
for name, K in (("Möbius strip", M), ("annulus", annulus)):
    print(f"{name:>13}: chi = {K.euler_characteristic}, orientable = {is_orientable(K)}, boundary circles = {boundary_components(K)}")

# %%
# A walk along the centre line comes back upside down
# ---------------------------------------------------
#
# Carrying the normal vector of the strip once around the centre line
# returns it pointing the other way; for the annulus it returns unchanged.

u = np.linspace(0, 2 * np.pi, 13)
fig = plt.figure(figsize=(12, 5))
for i, (name, K, twist) in enumerate((("Möbius strip", M, 0.5), ("annulus", annulus, 0.0))):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d", computed_zorder=False)
    plot_complex(K, ax=ax, alpha=0.3, vertex_size=0)
    if twist:
        center = np.column_stack([np.cos(u), np.sin(u), 0 * u])
        normal = np.column_stack([-np.sin(twist * u) * np.cos(u), -np.sin(twist * u) * np.sin(u), np.cos(twist * u)])
    else:
        center = np.column_stack([2.0 * np.cos(u), 2.0 * np.sin(u), 1.0 + 0 * u])
        normal = np.column_stack([0 * u, 0 * u, 1 + 0 * u])
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(u)))
    ax.quiver(*center.T, *(0.5 * normal).T, colors=colors, lw=2)
    ax.set_title(f"{name}: orientable = {is_orientable(K)}")
    ax.view_init(elev=35, azim=-60)
    ax.set_axis_off()
fig.suptitle("Möbius (1865): a surface with one side cannot be oriented")
