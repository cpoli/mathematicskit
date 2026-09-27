r"""
The Courant-Isaacson-Rees upwind scheme
=======================================

Courant, Isaacson and Rees (1952) differenced :math:`u_x` on the side
the information comes *from*: for :math:`c > 0`,
:math:`u_j^{n+1} = u_j^n - \nu(u_j^n - u_{j-1}^n)`. The scheme is
stable for :math:`\nu \le 1`, exact at :math:`\nu = 1`, and otherwise
first-order accurate, with numerical diffusion
:math:`\tfrac12 c\,\Delta x(1-\nu)\,u_{xx}` that smears the pulse.
Differencing on the wrong side is unstable at every step size.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import AdvectionEquation1D

pulse = lambda x: np.exp(-200 * (x - 0.5) ** 2)

# %%
# Numerical diffusion shrinks as nu -> 1
# --------------------------------------

adv = AdvectionEquation1D(pulse, n=200, c=1.0)
fig, ax = plt.subplots()
ax.plot(adv.x, adv.exact(1.0), "k--", lw=2, label="exact")
for nu in (0.25, 0.5, 0.9, 1.0):
    sol = adv.solve(1.0, dt=nu * adv.dx, scheme="upwind")
    print(f"nu = {nu:4.2f}: peak {sol.final.max():.3f}, max error {np.max(np.abs(sol.final - adv.exact(1.0))):.2e}")
    ax.plot(sol.x, sol.final, label=f"upwind, nu = {nu}")
ax.set_xlabel("$x$")
ax.set_title("Upwind after one period: diffusion (1 - nu)")
ax.legend()

# %%
# The upwind side depends on the sign of c
# ----------------------------------------

left = AdvectionEquation1D(pulse, n=200, c=-1.0)
sol = left.solve(0.25, dt=0.8 * left.dx, scheme="upwind")
print(f"c = -1: max error after t = 0.25: {np.max(np.abs(sol.final - left.exact(0.25))):.2e}")

# %%
# First-order convergence
# -----------------------

print("   n    error   ratio")
previous = None
for n in (100, 200, 400, 800):
    a = AdvectionEquation1D(lambda x: np.sin(2 * np.pi * x), n=n)
    error = np.max(np.abs(a.solve(1.0, dt=0.5 * a.dx, scheme="upwind").final - a.exact(1.0)))
    print(f"{n:>4}  {error:.2e}  {previous / error if previous else float('nan'):5.2f}")
    previous = error

plt.show()
