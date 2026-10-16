r"""
Animated: the logistic map's cobweb as r grows
==============================================

A cobweb diagram iterates :math:`x_{k+1} = rx_k(1 - x_k)` graphically:
go up to the parabola, across to the diagonal, repeat. Sweeping
:math:`r` from 2.8 to 4 shows the period-doubling route to chaos as the
red web settles onto a fixed point, then a square (a period-2 cycle),
then nested squares (period 4, 8, ...), and from
:math:`r_\infty \approx 3.5699` a web that never closes. The bifurcation
diagram on the right marks where each frame sits.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.ode_dynamics.visualizers import animate_logistic_cobweb

# In Jupyter or JupyterLite, show animations as an HTML/JavaScript player
# (no ffmpeg needed).
plt.rcParams["animation.html"] = "jshtml"

# %%
# From a fixed point to chaos
# ---------------------------
# The sweep slows down through the period-doubling cascade, where the
# action is.

r_values = np.concatenate([np.linspace(2.8, 3.4, 12), np.linspace(3.4, 3.6, 24), np.linspace(3.6, 4.0, 14)[1:]])
anim = animate_logistic_cobweb(r_values, x0=0.2, n_transient=60, n_keep=60)
anim

# %%
plt.show()
