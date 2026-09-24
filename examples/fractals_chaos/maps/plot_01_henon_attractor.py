r"""
The Hénon attractor
=========================

Iterates Hénon's map (x, y) -> (1 - 1.4 x^2 + y, 0.3 x), zooms in on
its strange attractor to reveal repeated layers, and estimates its
fractal and Lyapunov characteristics.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import box_counting_dimension, henon_map

# %%
# The attractor and a zoom
# -----------------------------------------------------

orbit = henon_map(300000)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(*orbit.T, ",k", alpha=0.5)
axes[0].add_patch(plt.Rectangle((0.55, 0.15), 0.1, 0.05, fill=False, color="tab:red"))
axes[0].set_title("Hénon attractor")
zoom = orbit[(orbit[:, 0] > 0.55) & (orbit[:, 0] < 0.65) & (orbit[:, 1] > 0.15) & (orbit[:, 1] < 0.2)]
axes[1].plot(*zoom.T, ",k")
axes[1].set_title("zoom: the lines split into bundles of lines")

# %%
# Dimension and sensitivity
# -----------------------------------------------------

print(f"box-counting estimate {box_counting_dimension(orbit).dimension:.3f}; careful studies give about 1.26, and coarse box sizes run high")

a, b = 1.4, 0.3
x, y = 0.1, 0.1
v = np.array([1.0, 0.0])
total = 0.0
steps = 100000
for _ in range(steps):
    v = np.array([[-2 * a * x, 1.0], [b, 0.0]]) @ v
    x, y = 1 - a * x * x + y, b * x
    norm = np.linalg.norm(v)
    total += np.log(norm)
    v /= norm
print(f"largest Lyapunov exponent ~ {total / steps:.3f} (positive: chaotic)")
