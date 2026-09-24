r"""
Bendixson's negative criterion
====================================

If the divergence of a planar vector field keeps one sign on a simply
connected region, no closed orbit fits inside it. A damped oscillator
passes the test everywhere; the Van der Pol field fails it because its
divergence :math:`\mu(1 - x^2)` changes sign at :math:`|x| = 1` -- and
indeed its limit cycle must cross those lines.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.ode_dynamics.systems.limit_cycles import VanDerPolOscillator, bendixson_criterion

systems = {
    "damped oscillator": lambda x, y: (y, -x - 0.4 * y),
    "Van der Pol (mu=1)": lambda x, y: (y, (1.0 - x**2) * y - x),
}

# %%
# Divergence maps
# ---------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, f) in zip(axes, systems.items()):
    res = bendixson_criterion(f, (-3, 3), (-3, 3), n=201)
    lim = max(abs(res.divergence.min()), abs(res.divergence.max()))
    mesh = ax.pcolormesh(res.X, res.Y, res.divergence, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    fig.colorbar(mesh, ax=ax, label="divergence")
    ax.set_title(f"{name}: cycles ruled out = {res.rules_out_periodic_orbits}")
    print(f"{name}: rules out periodic orbits -> {res.rules_out_periodic_orbits}")

traj = VanDerPolOscillator([0.5, 0.0], mu=1.0).integrate((0.0, 40.0), dt=1e-3, method="rk4")
axes[1].plot(traj.y[:, 0], traj.y[:, 1], "k", lw=1)
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)
fig.tight_layout()

plt.show()
