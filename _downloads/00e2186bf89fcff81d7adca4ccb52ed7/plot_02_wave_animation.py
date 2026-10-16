r"""
Animated: a plucked string splitting, reflecting, and recombining
=================================================================

D'Alembert's solution :math:`u = F(x - ct) + G(x + ct)` says a plucked
string's bump splits into two half-height waves running in opposite
directions. At the fixed ends each reflects upside down; after one full
period :math:`2L/c` the string is back where it started. The leapfrog
solve (solid) runs at Courant number 1, where it reproduces d'Alembert's
formula (dashed) exactly at the grid points.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import WaveEquation1D, dalembert_solution
from mathematicskit.pde.visualizers import animate_solution

# In Jupyter or JupyterLite, show animations as an HTML/JavaScript player
# (no ffmpeg needed).
plt.rcParams["animation.html"] = "jshtml"


def pluck(x):
    return np.maximum(0.0, 1.0 - np.abs(x - 0.3) / 0.1)


string = WaveEquation1D(pluck, n=201, c=1.0)
sol = string.solve(2.0, dt=string.dx, method="leapfrog")

# %%
# One full period of the string
# -----------------------------

anim = animate_solution(sol, exact=lambda x, t: dalembert_solution(pluck, x, t, length=1.0), max_frames=81)
anim

# %%
plt.show()
