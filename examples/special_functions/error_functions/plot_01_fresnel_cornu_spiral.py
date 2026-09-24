r"""
Fresnel integrals and the Cornu spiral
===========================================

Draws the Euler-Cornu spiral (C(t), S(t)), which winds into the points
(1/2, 1/2) and (-1/2, -1/2), and the knife-edge diffraction intensity
that Fresnel computed from it.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import fresnel_integrals

# %%
# The spiral
# -----------------------------------------------------

t = np.linspace(-6, 6, 3000)
r = fresnel_integrals(t)
fig, ax = plt.subplots()
ax.plot(r.c, r.s, lw=1)
ax.plot([0.5, -0.5], [0.5, -0.5], "ro")
ax.set_aspect("equal")
ax.set_xlabel("C(t)")
ax.set_ylabel("S(t)")
ax.set_title("The Cornu spiral")
far = fresnel_integrals(1e4)
print(f"C(1e4) = {far.c:.6f}, S(1e4) = {far.s:.6f} (limit 1/2)")

# %%
# Diffraction at a straight edge
# -----------------------------------------------------
# Relative intensity past a knife edge at Fresnel parameter v:
# I/I0 = ((C(v) + 1/2)^2 + (S(v) + 1/2)^2) / 2.

v = np.linspace(-3, 5, 800)
f = fresnel_integrals(v)
intensity = 0.5 * ((f.c + 0.5) ** 2 + (f.s + 0.5) ** 2)
fig, ax = plt.subplots()
ax.plot(v, intensity)
ax.axhline(1.0, color="gray", ls=":")
ax.axvline(0.0, color="gray", ls=":")
ax.set_xlabel("Fresnel parameter v (shadow on the left)")
ax.set_ylabel(r"$I / I_0$")
ax.set_title("Knife-edge diffraction")
print(f"intensity at the geometric shadow edge: {0.5 * ((0.5) ** 2 + (0.5) ** 2):.2f}; peak: {intensity.max():.4f}")
