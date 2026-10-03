r"""
Animated: power iteration turning a vector towards the dominant eigenvector
===========================================================================

Each step of power iteration, :math:`v_{k+1} = A v_k / \|A v_k\|`,
stretches the component of :math:`v_k` along each eigenvector by its
eigenvalue. The component along the eigenvector of largest
:math:`|\lambda|` grows fastest, so :math:`v_k` swings round onto that
direction, the angle shrinking by the factor
:math:`|\lambda_2 / \lambda_1|` per step.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import power_iteration
from mathematicskit.linalg.visualizers import animate_power_iteration

# In Jupyter or JupyterLite, show animations as an HTML/JavaScript player
# (no ffmpeg needed).
plt.rcParams["animation.html"] = "jshtml"

# %%
# A 2x2 matrix in the plane
# -------------------------
# Eigenvalues 3 and 1.8, so the angle to the dominant eigenvector
# shrinks by :math:`1.8/3 = 0.6` per step. The start vector lies close to
# the other eigenvector, to give the iterate a long way to turn.

theta = np.deg2rad(30.0)
Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
A = Q @ np.diag([3.0, 1.8]) @ Q.T
result = power_iteration(A, tol=1e-12, max_iter=40, v0=Q[:, 1] + 0.02 * Q[:, 0])
print(f"lambda ~ {result.eigenvalues[0]:.10f} after {result.iterations} iterations")

anim = animate_power_iteration(A, result)
anim

# %%
plt.show()
