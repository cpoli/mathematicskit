r"""
The Lax-Friedrichs scheme: stabilizing centered differences
===========================================================

The centered FTCS scheme :math:`u_j^{n+1} = u_j^n - \tfrac{\nu}{2}(u_{j+1}^n - u_{j-1}^n)`
is unconditionally unstable for advection. Peter Lax (1954), following
Friedrichs, replaced :math:`u_j^n` by the neighbour average
:math:`\tfrac12(u_{j+1}^n + u_{j-1}^n)`. That one change makes the scheme
stable for :math:`\nu \le 1`, and its conservation form carries over to
nonlinear conservation laws and shocks -- at the price of heavy
numerical diffusion.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import AdvectionEquation1D, max_amplification

G_NAME = {"ftcs": "ftcs_advection"}  # amplification-factor name of each advection scheme

adv = AdvectionEquation1D(lambda x: np.exp(-200 * (x - 0.5) ** 2), n=200)

# %%
# FTCS blows up; averaging tames it
# ---------------------------------

fig, ax = plt.subplots()
ax.plot(adv.x, adv.exact(0.5), "k--", lw=2, label="exact")
for scheme in ("ftcs", "lax_friedrichs"):
    sol = adv.solve(0.5, dt=0.8 * adv.dx, scheme=scheme)
    print(f"{scheme:>15}: max |u| = {np.max(np.abs(sol.final)):.3g}, max |G| = {max_amplification(G_NAME.get(scheme, scheme), 0.8):.4f}")
    ax.plot(sol.x, np.clip(sol.final, -1.0, 2.0), label=scheme)
ax.set_ylim(-1.0, 2.0)
ax.set_xlabel("$x$")
ax.set_title("Same centered difference, with and without averaging (nu = 0.8)")
ax.legend()

# %%
# Lax-Friedrichs' numerical diffusion grows as nu shrinks
# -------------------------------------------------------
# Its modified equation has diffusion coefficient dx^2 (1 - nu^2) / (2 dt).

fig, ax = plt.subplots()
ax.plot(adv.x, adv.exact(1.0), "k--", lw=2, label="exact")
for nu in (0.3, 0.6, 0.95):
    sol = adv.solve(1.0, dt=nu * adv.dx, scheme="lax_friedrichs")
    ax.plot(sol.x, sol.final, label=f"nu = {nu}")
ax.set_xlabel("$x$")
ax.set_title("Lax-Friedrichs after one period")
ax.legend()

plt.show()
