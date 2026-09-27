r"""
Burgers' equation and the Hopf-Cole transformation
==================================================

J. M. Burgers (1948) proposed :math:`u_t + u\,u_x = \nu\,u_{xx}` as the
simplest model of nonlinear steepening balanced by viscosity. Eberhard
Hopf (1950) and Julian Cole (1951) independently found that
:math:`u = -2\nu\,\varphi_x/\varphi` turns it into the heat equation
:math:`\varphi_t = \nu\varphi_{xx}`, so it can be solved exactly. For
:math:`u_0 = \sin x`, :math:`\varphi_0 = e^{\cos x/(2\nu)}` has Fourier
coefficients :math:`I_n(1/(2\nu))`, giving the exact solution checked
here against the pseudo-spectral solver.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ive

from mathematicskit.pde import BurgersEquation1D

nu = 0.1


def hopf_cole_solution(x, t, nu, n_terms=80):
    """Exact solution for u0 = sin x: phi = I_0 + 2 sum I_n e^{-nu n^2 t} cos(n x)."""
    n = np.arange(1, n_terms + 1)[:, None]
    a = ive(n, 1.0 / (2 * nu)) * np.exp(-nu * n**2 * t)  # exponentially scaled; the scale cancels
    phi = ive(0, 1.0 / (2 * nu)) + 2.0 * np.sum(a * np.cos(n * x), axis=0)
    phi_x = -2.0 * np.sum(n * a * np.sin(n * x), axis=0)
    return -2.0 * nu * phi_x / phi


# %%
# Steepening toward a shock, then decay
# -------------------------------------

burgers = BurgersEquation1D(np.sin, n=128, nu=nu)
sol = burgers.solve(3.0, dt=1e-3, save_every=500)
fig, ax = plt.subplots()
for t, u in zip(sol.t, sol.u):
    exact = hopf_cole_solution(sol.x, t, nu)
    print(f"t = {t:.1f}: max |spectral - Hopf-Cole| = {np.max(np.abs(u - exact)):.1e}")
    line = ax.plot(sol.x, exact, lw=3, alpha=0.35)[0]
    ax.plot(sol.x, u, "--", color=line.get_color(), label=f"t = {t:.1f}")
ax.set_xlabel("$x$")
ax.set_title("Burgers: pseudo-spectral (dashed) vs. Hopf-Cole (thick)")
ax.legend()

# %%
# The transformation itself: the heat equation's solution phi
# -----------------------------------------------------------

x = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots()
for t in (0.0, 1.0, 3.0):
    n = np.arange(1, 81)[:, None]
    phi = ive(0, 5.0) + 2.0 * np.sum(ive(n, 5.0) * np.exp(-nu * n**2 * t) * np.cos(n * x), axis=0)
    ax.plot(x, phi, label=f"t = {t}")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\varphi$ (scaled by $e^{-5}$)")
ax.set_title(r"$\varphi$ solves the heat equation")
ax.legend()

plt.show()
