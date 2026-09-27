r"""
Fourier pseudo-spectral methods: Burgers' equation
==================================================

Steven Orszag (1969-1971) and Kreiss and Oliger (1972) made the case for
differentiating with the FFT instead of finite differences. For a
smooth periodic function the error falls faster than any power of the
grid spacing. Nonlinear terms are simply formed pointwise on the grid
("pseudo-spectral"). This script first compares Fourier and
finite-difference derivatives, then solves viscous Burgers' equation
:math:`u_t + u u_x = \nu u_{xx}` as a steepening wave, integrating the
spectral ODE system with :func:`mathematicskit.integrators.rk4_integrate`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import BurgersEquation1D, fourier_derivative
from mathematicskit.pde.visualizers import plot_snapshots

# %%
# Spectral versus finite-difference accuracy
# ------------------------------------------

f = lambda x: np.exp(np.sin(x))
df = lambda x: np.cos(x) * np.exp(np.sin(x))
ns = np.array([6, 8, 12, 16, 24, 32, 48, 64])
spectral, central = [], []
for n in ns:
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = x[1] - x[0]
    spectral.append(np.max(np.abs(fourier_derivative(f(x)) - df(x))))
    central.append(np.max(np.abs((np.roll(f(x), -1) - np.roll(f(x), 1)) / (2 * h) - df(x))))
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.semilogy(ns, np.maximum(spectral, 1e-17), "o-", label="Fourier (FFT)")
ax1.semilogy(ns, central, "s-", label="central difference, $O(h^2)$")
ax1.set_xlabel("grid points $N$")
ax1.set_ylabel("max derivative error")
ax1.set_title("Spectral accuracy: exponential convergence")
ax1.legend()
fig1.tight_layout()

# %%
# A steepening Burgers wave
# -------------------------

burgers = BurgersEquation1D(lambda x: np.sin(x), n=128, nu=0.02)
sol = burgers.solve(2.0, dt=0.5 * burgers.max_stable_dt(), save_every=40)
print(f"mean of u: initial {sol.u[0].mean():+.2e}, final {sol.final.mean():+.2e} (conserved)")
fig2, ax2 = plt.subplots(figsize=(7, 4))
plot_snapshots(sol, n_snapshots=6, ax=ax2)
ax2.set_title(r"Burgers: $u_t + u u_x = 0.02\,u_{xx}$, pseudo-spectral + RK4")
fig2.tight_layout()

plt.show()
