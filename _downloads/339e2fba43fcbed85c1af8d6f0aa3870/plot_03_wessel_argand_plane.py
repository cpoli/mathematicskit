r"""
Wessel and Argand's complex plane: multiplication rotates and scales
====================================================================

Caspar Wessel (1797) and Jean-Robert Argand (1806) drew :math:`a + bi`
as the point :math:`(a, b)`. Addition is then vector addition, and
multiplication by :math:`w = re^{i\varphi}` rotates every point by
:math:`\varphi` and scales it by :math:`r`. Multiplying by :math:`i` is
a quarter turn, so :math:`i^2 = -1` is a half turn.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import map_grid
from mathematicskit.complex_analysis.visualizers import plot_mapped_grid

# %%
# Powers of i: four quarter turns
# -------------------------------

fig, ax = plt.subplots(figsize=(5, 5))
z = 1.0 + 0.0j
for k in range(4):
    ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(1.12 * z.real - 0.05, 1.12 * z.imag - 0.05, f"$i^{k}$")
    z *= 1j
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), "k:", lw=0.8)
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal")
ax.set_title("Multiplying by i is a quarter turn")

# %%
# Multiplication by w = 1 + i: rotate 45 degrees, scale by sqrt 2
# ----------------------------------------------------------------

w = 1.0 + 1.0j
print(f"|w| = {abs(w):.6f}, arg w = {np.degrees(np.angle(w)):.1f} degrees")
grid = map_grid(lambda z: w * z, x_range=(0.0, 1.0), y_range=(0.0, 1.0), n_lines=6)
plot_mapped_grid(grid)

# %%
# Moduli multiply, arguments add
# ------------------------------

z1, z2 = 2.0 * np.exp(0.3j), 1.5 * np.exp(1.1j)
print(f"|z1 z2| = {abs(z1 * z2):.6f}  vs  |z1||z2| = {abs(z1) * abs(z2):.6f}")
print(f"arg(z1 z2) = {np.angle(z1 * z2):.6f}  vs  arg z1 + arg z2 = {np.angle(z1) + np.angle(z2):.6f}")

plt.show()
