r"""
Phase portraits and vector fields
=====================================

Overlays a nonlinear system's vector field (a quiver plot) with actual
integrated trajectories, using the classic nonlinear pendulum
:math:`\ddot\theta = -\sin\theta` as the example.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.ode_dynamics.systems.phase_portrait import Nonlinear2D, vector_field_grid
from mathkit.ode_dynamics.visualizers.plots import plot_phase_portrait, plot_vector_field

# %%
# Pendulum vector field and trajectories
# --------------------------------------------

X, Y, U, V = vector_field_grid(lambda theta, omega: (omega, -np.sin(theta)), (-3.5, 3.5), (-2.5, 2.5), n=20)

fig, ax = plt.subplots(figsize=(7, 5))
plot_vector_field(X, Y, U, V, ax=ax)

trajectories = []
for theta0, omega0 in [(0.5, 0.0), (2.0, 0.0), (3.0, 0.0), (0.0, 2.2)]:
    system = Nonlinear2D([theta0, omega0], f=lambda theta, omega: (omega, -np.sin(theta)))
    trajectories.append(system.integrate((0.0, 10.0), dt=1e-3, method="rk4"))

plot_phase_portrait(trajectories, ax=ax, color="firebrick", lw=1.2)
ax.set_title("Pendulum phase portrait: librating vs. rotating orbits")
fig.tight_layout()

plt.show()
