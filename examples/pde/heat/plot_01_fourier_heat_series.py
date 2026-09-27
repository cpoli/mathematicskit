r"""
Fourier's heat equation: a sine series whose modes decay
========================================================

Joseph Fourier solved :math:`u_t = \alpha u_{xx}` on a rod with ice-cold
ends by writing the initial temperature as a sine series. Each mode
:math:`\sin(k\pi x)` then evolves on its own, decaying like
:math:`e^{-\alpha (k\pi)^2 t}`: sharp features (high :math:`k`) vanish
almost instantly, leaving the smooth fundamental mode. This script
computes the sine coefficients of a square-wave start, watches the
series smooth out, and checks it against a finite-difference solve.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import HeatEquation1D, fourier_sine_coefficients, heat_series_solution


def square(x):
    return np.where((x > 0.25) & (x < 0.75), 1.0, 0.0)


# %%
# The initial temperature as a sine series
# ----------------------------------------

b = fourier_sine_coefficients(square, 60)
for k in range(1, 8):
    print(f"b_{k} = {b[k - 1]:+.4f}")

x = np.linspace(0.0, 1.0, 400)
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(x, square(x), color="black", lw=2, label="initial temperature")
for n_terms, color in ((1, "goldenrod"), (5, "darkorange"), (60, "firebrick")):
    ax1.plot(x, heat_series_solution(b[:n_terms], x, t=0.0), color=color, label=f"{n_terms} sine terms")
ax1.set_xlabel("$x$")
ax1.set_title("A square wave as a sum of sine modes")
ax1.legend(fontsize=8)
fig1.tight_layout()

# %%
# High modes die first
# --------------------
# The :math:`k`-th mode's decay rate grows like :math:`k^2`, so after
# a short time only the first mode is visible.

fig2, ax2 = plt.subplots(figsize=(7, 4))
for t, color in zip((0.0, 0.002, 0.01, 0.05, 0.2), plt.get_cmap("viridis")(np.linspace(0, 0.9, 5)), strict=False):
    ax2.plot(x, heat_series_solution(b, x, t), color=color, label=f"t = {t}")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$u$")
ax2.set_title("Fourier's series solution smooths the square wave")
ax2.legend(fontsize=8)
fig2.tight_layout()

# %%
# The series agrees with a finite-difference solve
# ------------------------------------------------

heat = HeatEquation1D(square, n=201)
sol = heat.solve_theta(0.05, dt=1e-4, theta=0.5)
series = heat_series_solution(b, sol.x, 0.05)
print(f"max |finite difference - Fourier series| at t = 0.05: {np.max(np.abs(sol.final - series)):.2e}")

plt.show()
