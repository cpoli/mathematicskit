r"""
Gauss's linking number (1833)
=============================

Gauss's double integral
:math:`\frac{1}{4\pi}\oint\oint \frac{(r_1 - r_2)\cdot(dr_1 \times dr_2)}{|r_1 - r_2|^3}`
counts how many times one closed curve winds around another. It is an
integer that no deformation can change without passing one curve
through the other. Below: the unlink (0), the Hopf link (:math:`\pm 1`,
the sign set by the orientations), and torus links whose components
wind :math:`n` times around each other.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import linking_number

t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
ring = np.column_stack([np.cos(t), np.sin(t), 0 * t])
links = {
    "unlink": (ring, np.column_stack([3 + np.cos(t), 0 * t, np.sin(t)])),
    "Hopf link": (ring, np.column_stack([1 + np.cos(t), 0 * t, np.sin(t)])),
    "Hopf link, reversed": (ring, np.column_stack([1 + np.cos(t), 0 * t, np.sin(t)])[::-1]),
}


def torus_link(n, phase):
    v = n * t + phase
    return np.column_stack([(2 + np.cos(v)) * np.cos(t), (2 + np.cos(v)) * np.sin(t), np.sin(v)])


for n in (1, 2, 3):
    links[f"(2, {2 * n}) torus link"] = (torus_link(n, 0.0), torus_link(n, np.pi))

for name, (a, b) in links.items():
    print(f"{name:>20}: linking number {linking_number(a, b):+.6f}")

# %%
# The links
# ---------

fig = plt.figure(figsize=(12, 7))
for i, (name, (a, b)) in enumerate(links.items()):
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    for curve, color in ((a, "tab:blue"), (b, "tab:orange")):
        closed = np.vstack([curve, curve[:1]])
        ax.plot(*closed.T, color=color, lw=2)
    ax.set_title(f"{name}: lk = {round(linking_number(a, b))}")
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_axis_off()
fig.suptitle("Gauss (1833): the linking integral is an integer")

# %%
# The integrand is a solid angle
# ------------------------------
#
# Gauss read the integral as the solid angle swept by the direction from
# one curve to the other. Refining the polygons barely changes the sum,
# which stays an integer: it is exact for polygons, not an approximation.

ns = [12, 24, 48, 96, 192]
values = []
for n in ns:
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values.append(linking_number(np.column_stack([np.cos(s), np.sin(s), 0 * s]), np.column_stack([1 + np.cos(s), 0 * s, np.sin(s)])))
print("Hopf link with n-gon curves:", np.round(values, 12))
