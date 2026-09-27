r"""
Schwarz's lemma: self-maps of the disk fixing 0 cannot expand
=============================================================

If :math:`f` maps the unit disk into itself with :math:`f(0) = 0`, then
:math:`|f(z)| \le |z|` and :math:`|f'(0)| \le 1`, with equality only
for rotations :math:`f(z) = e^{i\theta}z` (Schwarz 1869; Carathéodory
gave the general statement in 1912). Blaschke products, the model
self-maps of the disk, obey it with room to spare.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import complex_derivative, complex_grid, mobius_transform


def blaschke_factor(a):
    """The disk automorphism (z - a) / (1 - conj(a) z), a Möbius transformation."""
    return lambda z: mobius_transform(z, 1, -a, -np.conj(a), 1)


maps = {
    "rotation e^{0.7i} z": lambda z: np.exp(0.7j) * z,
    "z^2": lambda z: z**2,
    "z (z - 0.5)/(1 - 0.5 z)": lambda z: z * blaschke_factor(0.5)(z),
    "z (z - 0.3i)/(1 + 0.3i z)": lambda z: z * blaschke_factor(0.3j)(z),
}

# %%
# :math:`|f(z)| \le |z|` everywhere in the disk, :math:`|f'(0)| \le 1`
# ----------------------------------------------------------------------

z = complex_grid((-0.99, 0.99), (-0.99, 0.99), 201)
z = z[(np.abs(z) < 0.99) & (z != 0)]
fig, ax = plt.subplots()
for name, f in maps.items():
    ratio = np.abs(f(z)) / np.abs(z)
    print(f"{name:26s} max |f(z)|/|z| = {ratio.max():.6f},  |f'(0)| = {abs(complex_derivative(f, 0.0)):.6f}")
    order = np.argsort(np.abs(z))
    ax.plot(np.abs(z)[order][::50], ratio[order][::50], ".", ms=3, label=name)
ax.axhline(1, color="k", lw=1)
ax.set_xlabel("|z|")
ax.set_ylabel("|f(z)| / |z|")
ax.set_title("Schwarz's lemma: the ratio never exceeds 1")
ax.legend(fontsize=8)
