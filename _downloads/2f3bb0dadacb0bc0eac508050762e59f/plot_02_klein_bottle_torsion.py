r"""
Klein's bottle and torsion in homology (1882)
=============================================

Felix Klein's bottle glues a cylinder's ends with a reflection. It is a
closed surface with one side and no edge. It cannot sit in
:math:`\mathbb{R}^3` without crossing itself, as the figure-8 immersion
below does. Its first homology group is :math:`\mathbb{Z} \oplus
\mathbb{Z}/2`: a loop that is not a boundary but whose double is. The
:math:`\mathbb{Z}/2` torsion hides from rational Betti numbers and shows
up mod 2.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.topology import betti_numbers, homology, is_orientable, klein_bottle, torus
from mathematicskit.topology.visualizers import plot_complex

K = klein_bottle(40, 24)
T = torus(6, 4)
for name, X in (("torus", T), ("Klein bottle", klein_bottle(6, 4))):
    print(f"{name:>12}: {homology(X)}; orientable = {is_orientable(X)}")
    print(f"{'':>12}  Betti over Q: {betti_numbers(X)}, over Z/2: {betti_numbers(X, field=2)}, over Z/3: {betti_numbers(X, field=3)}")

# %%
# The figure-8 immersion
# ----------------------

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
heights = K.coordinates[:, 2]
plot_complex(K, ax=ax, face_values=[heights[list(t)].mean() for t in K.simplices(2)], alpha=0.85, vertex_size=0)
ax.view_init(elev=30, azim=-50)
ax.set_axis_off()
ax.set_title(r"Klein bottle: $H_1 = \mathbb{Z} \oplus \mathbb{Z}/2$")
