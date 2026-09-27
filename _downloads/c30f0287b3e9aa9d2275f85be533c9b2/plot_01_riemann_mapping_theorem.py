r"""
Riemann's mapping theorem: the upper half-plane mapped conformally onto the disk
================================================================================

Riemann (1851) asserted that every simply connected proper subdomain of
the plane can be mapped conformally onto the unit disk. For the upper
half-plane the map is explicit: the Cayley transform
:math:`w = (z - i)/(z + i)`, a Möbius transformation. The image of the
Cartesian grid is a family of circles that still meet at right angles,
because conformal maps preserve angles.
"""

# %%
import numpy as np

from mathematicskit.complex_analysis import cauchy_riemann, map_grid, mobius_transform
from mathematicskit.complex_analysis.visualizers import plot_mapped_grid

cayley = lambda z: mobius_transform(z, 1, -1j, 1, 1j)

# %%
# The grid and its image
# ----------------------

grid = map_grid(cayley, (-4, 4), (0, 4), n_lines=17, n_points=400)
axes = plot_mapped_grid(grid)
axes[0].set_title("upper half-plane")
axes[1].set_title(r"unit disk: $w = (z - i)/(z + i)$")
t = np.linspace(0, 2 * np.pi, 200)
axes[1].plot(np.cos(t), np.sin(t), "k", lw=1)

# %%
# Conformality: holomorphic with nonzero derivative, and angle preserving
# -----------------------------------------------------------------------

rng = np.random.default_rng(1)
samples = rng.normal(size=5) + 1j * rng.uniform(0.1, 3, size=5)
print("max |w| over upper half-plane samples:", np.max(np.abs(cayley(samples))))
print("max Cauchy-Riemann residual:", max(cauchy_riemann(cayley, z).residual for z in samples))
h = 1e-6
for z in samples[:3]:
    d1, d2 = cayley(z + h) - cayley(z), cayley(z + 1j * h) - cayley(z)
    print(f"z = {z:.2f}: image angle between horizontal and vertical directions = {np.degrees(np.angle(d2 / d1)):.6f} deg")
