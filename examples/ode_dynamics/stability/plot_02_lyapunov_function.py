r"""
Lyapunov's direct method
==============================

For a stable linear system :math:`\dot x = Ax`, solving the Lyapunov
equation :math:`A^T P + PA = -Q` gives an energy-like function
:math:`V(x) = x^T P x` whose level sets are ellipses that every
trajectory crosses inward -- a proof of stability that never solves the
ODE itself.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.ode_dynamics.systems.phase_portrait import Linear2D
from mathematicskit.ode_dynamics.systems.stability import lyapunov_quadratic_form

A = np.array([[-0.3, 2.0], [-1.0, -0.3]])
res = lyapunov_quadratic_form(A)
print("P =\n", np.round(res.P, 4))
print("P positive definite (origin asymptotically stable):", res.positive_definite)

# %%
# Trajectories cross the level sets of V inward
# ---------------------------------------------------

xs = np.linspace(-2.5, 2.5, 200)
X, Y = np.meshgrid(xs, xs)
V = res.P[0, 0] * X**2 + 2 * res.P[0, 1] * X * Y + res.P[1, 1] * Y**2
traj = Linear2D([2.0, 1.0], A=A).integrate((0.0, 20.0), dt=1e-2, method="rk4")
V_traj = np.einsum("ti,ij,tj->t", traj.y, res.P, traj.y)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].contour(X, Y, V, levels=12, cmap="viridis")
axes[0].plot(traj.y[:, 0], traj.y[:, 1], "k", lw=1)
axes[0].set_aspect("equal")
axes[0].set_title("Level sets of V(x) = x^T P x and a trajectory")
axes[1].semilogy(traj.t, V_traj)
axes[1].set_xlabel("t")
axes[1].set_ylabel("V(x(t))")
axes[1].set_title("V decreases monotonically along the flow")
fig.tight_layout()

# %%
# An unstable matrix admits no positive-definite P
# -------------------------------------------------------

print("unstable spiral, P positive definite:", lyapunov_quadratic_form([[0.1, 1.0], [-1.0, 0.1]]).positive_definite)

plt.show()
