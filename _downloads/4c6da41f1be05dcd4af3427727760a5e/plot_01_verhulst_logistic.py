r"""
Verhulst's logistic growth
================================

Integrating :math:`\dot N = rN(1 - N/K)` numerically reproduces
Verhulst's closed-form sigmoid exactly: growth looks exponential at
first, is fastest at :math:`N = K/2`, and saturates at the carrying
capacity :math:`K` from any positive starting population.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.ode_dynamics.systems.population import LogisticGrowth, logistic_growth_solution

# %%
# Numerical integration versus the closed form
# ---------------------------------------------------

r, K = 0.8, 100.0
t = np.linspace(0.0, 15.0, 400)
fig, ax = plt.subplots(figsize=(7, 4.5))
for n0 in (2.0, 20.0, 150.0):
    traj = LogisticGrowth([n0], r=r, K=K).integrate((0.0, 15.0), dt=1e-2, method="rk4")
    ax.plot(traj.t, traj.y[:, 0], lw=3, alpha=0.4, label=f"RK4, N0={n0:g}")
    ax.plot(t, logistic_growth_solution(t, n0, r, K), "k--", lw=1)
    err = np.max(np.abs(traj.y[:, 0] - logistic_growth_solution(traj.t, n0, r, K)))
    print(f"N0={n0:6.1f}: max |RK4 - exact| = {err:.2e}")
ax.axhline(K, color="gray", lw=0.8)
ax.set_xlabel("t")
ax.set_ylabel("N(t)")
ax.set_title("Logistic growth approaches the carrying capacity K (dashed: exact)")
ax.legend()
fig.tight_layout()

plt.show()
