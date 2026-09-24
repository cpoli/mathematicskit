r"""
Fatou and Julia: Julia sets of z -> z^2 + c
=================================================

Iterates the quadratic map :math:`z \mapsto z^2 + c` from every starting
point :math:`z_0` of a grid. Points whose orbits stay bounded form the
filled Julia set; its boundary, the Julia set, separates the Fatou set,
where nearby orbits behave alike, from the region where orbits escape
to infinity. Changing :math:`c` turns the Julia set from a circle into
a fractal curve or a disconnected dust.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import julia_set
from mathematicskit.fractals_chaos.visualizers.plots import plot_escape_time

# %%
# Four values of c
# ----------------
#
# For :math:`c = 0` the Julia set is the unit circle. The other three are
# fractal: the "basilica" (:math:`c=-1`) and Douady's "rabbit" are
# connected, while for :math:`c` outside the Mandelbrot set the Julia set
# breaks up into a totally disconnected dust.

params = {
    "c = 0 (circle)": 0.0 + 0.0j,
    "c = -1 (basilica)": -1.0 + 0.0j,
    "c = -0.123 + 0.745i (rabbit)": -0.123 + 0.745j,
    "c = 0.4 + 0.4i (dust)": 0.4 + 0.4j,
}
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for ax, (label, c) in zip(axes.ravel(), params.items()):
    result = julia_set(c=c, extent=(-1.6, 1.6, -1.6, 1.6), resolution=300, max_iter=200)
    plot_escape_time(result, ax=ax, cmap="magma")
    ax.set_title(label)
    bounded = np.mean(result.iterations == result.max_iter)
    print(f"{label:30s} fraction of grid with bounded orbits: {bounded:.3f}")
fig.tight_layout()

# %%
# Sensitive dependence near the Julia set
# ---------------------------------------
#
# Two starting points a distance :math:`10^{-6}` apart, straddling the
# unit circle (the Julia set for :math:`c = 0`), end up in different
# places: one orbit tends to 0, the other escapes. Two points in the same
# Fatou component stay together.

for z0 in (1.0 - 5e-7, 1.0 + 5e-7, 0.5, 0.5 + 1e-6):
    z = complex(z0)
    for _ in range(40):
        z = z * z
        if abs(z) > 1e6:
            break
    print(f"z0 = {z0:.7f}: |z_n| -> {'infinity' if abs(z) > 1e6 else f'{abs(z):.1e}'}")
