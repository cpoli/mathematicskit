r"""
Gauss's Theorema Egregium: curvature you can measure from inside
======================================================================

Computes the Gaussian curvature K of a sphere, a cylinder, and a torus.
The cylinder bends in space but has K = 0, because it can be unrolled
flat without stretching, while no piece of a sphere can. Integrating K
over the whole surface gives 4 pi for the sphere and 0 for the torus,
as the Gauss-Bonnet theorem predicts.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from mathematicskit.geometry import cylinder_surface, sphere_surface, surface_curvature, torus_surface

# %%
# Curvature of three surfaces
# -----------------------------------------------------

cases = {
    "sphere (R = 1)": surface_curvature(sphere_surface(1.0), np.linspace(1e-3, np.pi - 1e-3, 201), np.linspace(0, 2 * np.pi, 201)),
    "cylinder (R = 1)": surface_curvature(cylinder_surface(1.0), np.linspace(-1, 1, 101), np.linspace(0, 2 * np.pi, 201)),
    "torus (R = 2, r = 0.7)": surface_curvature(torus_surface(2.0, 0.7), np.linspace(0, 2 * np.pi, 201), np.linspace(0, 2 * np.pi, 201)),
}
fig = plt.figure(figsize=(13, 4))
for k, (name, result) in enumerate(cases.items(), start=1):
    ax = fig.add_subplot(1, 3, k, projection="3d")
    K = result.gaussian
    colors = plt.cm.coolwarm((K - K.min()) / (np.ptp(K) + 1e-12)) if np.ptp(K) > 1e-9 else plt.cm.coolwarm(np.full(K.shape, 0.5))
    ax.plot_surface(*result.points, facecolors=colors, rstride=4, cstride=4, linewidth=0)
    ax.set_title(f"{name}\nK from {K.min():+.3f} to {K.max():+.3f}", fontsize=9)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")

# %%
# Total curvature (Gauss-Bonnet)
# -----------------------------------------------------

for name, result in cases.items():
    if name.startswith("cylinder"):
        continue
    total = integrate.trapezoid(integrate.trapezoid(result.gaussian * result.area_element, result.v, axis=1), result.u)
    print(f"{name:22s}: integral of K dA = {total:+.5f}")
print(f"{'4 pi':22s}  = {4 * np.pi:+.5f}")
