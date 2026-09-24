r"""
Diffusion-limited aggregation
===================================

Grows a cluster by releasing random walkers that stick on first contact,
and measures its fractal dimension from the mass-radius relation
N(r) ~ r^D.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import dla_cluster

# %%
# Grow and draw the cluster
# -----------------------------------------------------

cluster = dla_cluster(12000, seed=0)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(*cluster.T, c=np.arange(len(cluster)), cmap="viridis", s=0.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"{len(cluster)} particles, colored by arrival time")

# %%
# Mass-radius scaling
# -----------------------------------------------------

r = np.sqrt((cluster**2).sum(axis=1))
radii = np.logspace(np.log10(5), np.log10(r.max() / 2), 12)
mass = np.array([(r < rho).sum() for rho in radii])
slope = np.polyfit(np.log(radii), np.log(mass), 1)[0]
print(f"mass-radius dimension D = {slope:.3f} (Witten and Sander: about 1.7)")

fig, ax = plt.subplots()
ax.loglog(radii, mass, "o", label="particles within radius r")
ax.loglog(radii, mass[0] * (radii / radii[0]) ** slope, "--", label=f"slope {slope:.2f}")
ax.loglog(radii, mass[0] * (radii / radii[0]) ** 2, ":", label="slope 2 (compact disk)")
ax.set_xlabel("r")
ax.set_ylabel("N(r)")
ax.legend()
