r"""
Von Neumann stability analysis: one Fourier mode at a time
==========================================================

John von Neumann's method substitutes a single Fourier mode
:math:`u_j^n = G^n e^{ij\xi}` into a linear scheme. Each step multiplies
the mode by the *amplification factor* :math:`G(\xi)`, so the scheme is
stable exactly when :math:`|G(\xi)| \le 1` for every wavenumber. This
script plots :math:`|G|` for heat and advection schemes, recovers the
FTCS limit :math:`r \le 1/2` numerically, and shows why FTCS advection
is unstable for *every* time step.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import max_amplification
from mathematicskit.pde.visualizers import plot_amplification_factors

# %%
# Heat-equation schemes
# ---------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_amplification_factors(["ftcs_heat", "crank_nicolson_heat", "backward_euler_heat"], 0.6, ax=axes[0])
axes[0].set_title("Heat equation, diffusion number r = 0.6")

# %%
# Advection schemes
# -----------------

plot_amplification_factors(["upwind", "lax_friedrichs", "lax_wendroff", "ftcs_advection"], 0.8, ax=axes[1])
axes[1].set_title("Advection, Courant number 0.8")
fig.tight_layout()

# %%
# Finding the stability limits numerically
# ----------------------------------------

rs = np.linspace(0.3, 0.7, 401)
limit = rs[np.argmax([max_amplification("ftcs_heat", r) > 1 + 1e-12 for r in rs]) - 1]
print(f"FTCS heat is stable up to r = {limit:.3f} (theory: 1/2)")
for nu in (0.01, 0.5, 1.0):
    print(f"FTCS advection, nu = {nu}: max |G| = {max_amplification('ftcs_advection', nu):.6f}  (> 1 for every nu > 0)")
for r in (1.0, 100.0):
    print(f"Crank-Nicolson, r = {r}: max |G| = {max_amplification('crank_nicolson_heat', r):.3f}")

plt.show()
