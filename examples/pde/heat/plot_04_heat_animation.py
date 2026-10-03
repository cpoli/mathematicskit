r"""
Animated: the heat equation smoothing a rough initial profile
=============================================================

Under :math:`u_t = \alpha u_{xx}` with :math:`u(0) = u(1) = 0`, the
Fourier mode :math:`\sin(n\pi x)` decays like
:math:`e^{-\alpha n^2 \pi^2 t}`: the higher the frequency, the faster it
dies. Watching a jagged profile evolve shows exactly that: the wiggles
vanish almost at once, then the remaining hump sinks slowly as the
single slowest mode. The dashed curve is Fourier's series solution.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde import HeatEquation1D, fourier_sine_coefficients, heat_series_solution
from mathematicskit.pde.visualizers import animate_solution

# In Jupyter or JupyterLite, show animations as an HTML/JavaScript player
# (no ffmpeg needed).
plt.rcParams["animation.html"] = "jshtml"


def u0(x):
    return np.sin(np.pi * x) + 0.4 * np.sin(5 * np.pi * x) + 0.25 * np.sin(13 * np.pi * x)


heat = HeatEquation1D(u0, n=101)
sol = heat.solve_theta(0.15, dt=5e-4, theta=0.5)  # Crank-Nicolson
coefficients = fourier_sine_coefficients(u0, n_terms=20)

# %%
# Fast modes die first
# --------------------
# The :math:`\sin(13\pi x)` wiggles fade within :math:`t \approx 0.001`
# and :math:`\sin(5\pi x)` by :math:`t \approx 0.01`, so the frames are
# spaced geometrically in time to catch both.

frames = np.concatenate([[0], np.geomspace(1, len(sol.t) - 1, 70)])
anim = animate_solution(sol, exact=lambda x, t: heat_series_solution(coefficients, x, t), frames=frames, interval=90)
anim

# %%
plt.show()
