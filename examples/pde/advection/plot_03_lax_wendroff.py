r"""
The Lax-Wendroff scheme: second order, and dispersive wiggles
=============================================================

Lax and Wendroff (1960) used the PDE itself to replace time derivatives
in a Taylor expansion, :math:`u_{tt} = c^2 u_{xx}`, giving
:math:`u_j^{n+1} = u_j^n - \tfrac{\nu}{2}(u_{j+1}^n - u_{j-1}^n) + \tfrac{\nu^2}{2}(u_{j+1}^n - 2u_j^n + u_{j-1}^n)`.
It is second-order accurate and far less diffusive than upwind on smooth
waves, but, as Godunov later proved every linear second-order scheme
must, it oscillates near discontinuities.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import AdvectionEquation1D

# %%
# A smooth wave after ten periods
# -------------------------------

adv = AdvectionEquation1D(lambda x: np.sin(2 * np.pi * x), n=50)
fig, ax = plt.subplots()
ax.plot(adv.x, adv.exact(10.0), "k--", lw=2, label="exact")
for scheme in ("upwind", "lax_wendroff"):
    sol = adv.solve(10.0, dt=0.5 * adv.dx, scheme=scheme)
    print(f"smooth, {scheme:>12}: max error {np.max(np.abs(sol.final - adv.exact(10.0))):.3f}")
    ax.plot(sol.x, sol.final, "o-", ms=3, label=scheme)
ax.set_xlabel("$x$")
ax.set_title("Ten periods on 50 points: second order keeps the amplitude")
ax.legend()

# %%
# A square wave: Lax-Wendroff overshoots, upwind smears
# -----------------------------------------------------

square = AdvectionEquation1D(lambda x: np.where((x > 0.3) & (x < 0.6), 1.0, 0.0), n=200)
fig, ax = plt.subplots()
ax.plot(square.x, square.exact(1.0), "k--", lw=2, label="exact")
for scheme in ("upwind", "lax_wendroff"):
    sol = square.solve(1.0, dt=0.5 * square.dx, scheme=scheme)
    print(f"square, {scheme:>12}: min {sol.final.min():+.3f}, max {sol.final.max():.3f}")
    ax.plot(sol.x, sol.final, label=scheme)
ax.set_xlabel("$x$")
ax.set_title("Dispersive oscillations at discontinuities")
ax.legend()

plt.show()
