r"""
The zoo of 2D linear fixed points
=====================================

Every 2D linear system's qualitative phase portrait near the origin is
fixed by its Jacobian's trace and determinant: node, saddle, spiral, or
center. This script classifies six canonical matrices and plots their
phase portraits side by side.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.ode_dynamics.systems.phase_portrait import Linear2D
from mathkit.ode_dynamics.systems.stability import classify_fixed_point_2d

# %%
# Six canonical linear systems
# --------------------------------

matrices = {
    "stable node": np.array([[-1.0, 0.0], [0.0, -2.0]]),
    "unstable node": np.array([[1.0, 0.0], [0.0, 2.0]]),
    "saddle": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "stable spiral": np.array([[-0.2, 1.0], [-1.0, -0.2]]),
    "unstable spiral": np.array([[0.2, 1.0], [-1.0, 0.2]]),
    "center": np.array([[0.0, 1.0], [-1.0, 0.0]]),
}

fig, axes = plt.subplots(2, 3, figsize=(11, 7))
rng = np.random.default_rng(0)
for ax, (name, A) in zip(axes.ravel(), matrices.items()):
    result = classify_fixed_point_2d(A)
    print(f"{name:>16s}: eigenvalues={np.round(result.eigenvalues, 3)}, classified as '{result.classification}'")
    for _ in range(6):
        state0 = rng.uniform(-1, 1, 2)
        system = Linear2D(state0, A=A)
        traj = system.integrate((0.0, 8.0), dt=1e-3, method="rk4")
        ax.plot(traj.y[:, 0], traj.y[:, 1], lw=0.8)
    ax.plot(0, 0, "ko", ms=4)
    ax.set_title(name)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
fig.tight_layout()

plt.show()
