r"""
The Courant-Friedrichs-Lewy condition
=====================================

Courant, Friedrichs and Lewy (1928) observed that an explicit scheme
for :math:`u_t + c\,u_x = 0` computes each new value from a few
neighbors, so information can travel at most one cell per step. If the
true wave moves farther than that -- Courant number
:math:`\nu = c\,\Delta t/\Delta x > 1` -- the scheme cannot possibly see
where the solution came from, and it diverges. This script advects a
pulse with the upwind scheme below and above :math:`\nu = 1`,
and with Lax-Wendroff, whose limit is the same.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import AdvectionEquation1D, check_cfl

adv = AdvectionEquation1D(lambda x: np.exp(-200 * (x - 0.5) ** 2), n=200, c=1.0)

# %%
# Below and above nu = 1
# ----------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, scheme in zip(axes, ("upwind", "lax_wendroff"), strict=False):
    ax.plot(adv.x, adv.exact(1.0), "k--", lw=2, label="exact (one period)")
    for nu, color in ((0.5, "tab:blue"), (0.95, "tab:green"), (1.2, "tab:red")):
        sol = adv.solve(1.0, dt=nu * adv.dx, scheme=scheme)
        check = sol.extra["stability"]
        print(f"{scheme:>12}  nu = {nu:4.2f}  stable = {check.stable!s:5}  max |u| = {np.max(np.abs(sol.final)):.3g}")
        ax.plot(sol.x, np.clip(sol.final, -0.5, 1.5), color=color, label=f"nu = {nu}")
    ax.set_title(scheme)
    ax.set_xlabel("$x$")
    ax.set_ylim(-0.5, 1.5)
axes[0].legend(fontsize=8)
fig.suptitle("CFL: the Courant number must not exceed 1")
fig.tight_layout()

# %%
# The domain-of-dependence argument, in numbers
# ---------------------------------------------

for dt in (0.004, 0.005, 0.006):
    r = check_cfl(adv.c, dt, adv.dx, "upwind")
    print(f"dt = {dt}: Courant number {r.number:.2f} (limit {r.limit:.0f}) -> {'stable' if r.stable else 'UNSTABLE'}")

plt.show()
