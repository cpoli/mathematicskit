r"""
FitzHugh-Nagumo: excitability and spiking
===============================================

With no injected current the model neuron rests at a stable fixed point.
Between the two Hopf currents that fixed point is unstable, and the
neuron fires repetitively on a relaxation limit cycle that runs along
the outer branches of the cubic :math:`v`-nullcline.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.ode_dynamics.systems.excitable import FitzHughNagumo, fitzhugh_nagumo_hopf_currents

i_low, i_high = fitzhugh_nagumo_hopf_currents()
print(f"rest state unstable for {i_low:.4f} < I < {i_high:.4f}")

# %%
# Phase plane and voltage traces
# ------------------------------------

v = np.linspace(-2.5, 2.5, 300)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for current, color in ((0.0, "tab:blue"), (0.5, "tab:red")):
    system = FitzHughNagumo([-1.0, 1.0], current=current)
    traj = system.integrate((0.0, 200.0), dt=1e-2, method="rk4")
    axes[0].plot(traj.y[:, 0], traj.y[:, 1], color=color, lw=1, label=f"I={current}")
    axes[0].plot(v, v - v**3 / 3 + current, color=color, ls=":", lw=1)
    axes[1].plot(traj.t, traj.y[:, 0], color=color, label=f"I={current}")
axes[0].plot(v, (v + system.a) / system.b, "k--", lw=1, label="w-nullcline")
axes[0].set_ylim(-1, 2.5)
axes[0].set_xlabel("v")
axes[0].set_ylabel("w")
axes[0].set_title("Nullclines (dotted: v-nullcline) and orbits")
axes[0].legend()
axes[1].set_xlabel("t")
axes[1].set_ylabel("v")
axes[1].set_title("Rest (I=0) vs. periodic spiking (I=0.5)")
axes[1].legend()
fig.tight_layout()

plt.show()
