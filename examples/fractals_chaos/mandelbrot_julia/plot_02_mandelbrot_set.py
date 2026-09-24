r"""
The Mandelbrot set
========================

Colours every parameter :math:`c` by how quickly the orbit of
:math:`z_0 = 0` under :math:`z \mapsto z^2 + c` escapes. The black
region, the Mandelbrot set, is the set of :math:`c` whose orbit stays
bounded -- equivalently, whose Julia set is connected. Zooming into its
boundary reveals ever finer detail, including small copies of the whole
set.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.fractals_chaos import mandelbrot_set
from mathematicskit.fractals_chaos.visualizers.plots import plot_escape_time

# %%
# The whole set
# -------------

mandelbrot = mandelbrot_set(resolution=400, max_iter=200)
ax = plot_escape_time(mandelbrot)
ax.set_title("Mandelbrot set")

# %%
# Zooming into the boundary
# -------------------------
#
# Near :math:`c \approx -1.77` on the real axis, a small copy of the
# Mandelbrot set sits inside the boundary's filaments.

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
seahorse = mandelbrot_set(extent=(-0.80, -0.70, 0.05, 0.15), resolution=300, max_iter=400)
plot_escape_time(seahorse, ax=axes[0], cmap="twilight")
axes[0].set_title("Seahorse valley")
minibrot = mandelbrot_set(extent=(-1.80, -1.74, -0.03, 0.03), resolution=300, max_iter=400)
plot_escape_time(minibrot, ax=axes[1], cmap="twilight")
axes[1].set_title("A small copy near c = -1.77")
fig.tight_layout()

# %%
# Membership check
# ----------------
#
# :math:`c=-1` and :math:`c=0.25` lie in the set (bounded orbits);
# :math:`c = 0.3` and :math:`c = 1` do not.

for c in (-1.0, 0.25, 0.3, 1.0):
    z, steps = 0.0, 0
    while abs(z) <= 2.0 and steps < 1000:
        z, steps = z * z + c, steps + 1
    print(f"c = {c:5.2f}: {'bounded (in the set)' if abs(z) <= 2.0 else f'escapes after {steps} steps'}")
