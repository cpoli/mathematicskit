r"""
The logistic map's period-doubling route to chaos
=======================================================

As the growth-rate parameter ``r`` increases, the logistic map's
long-time attractor bifurcates from a fixed point to a period-2 cycle,
then period-4, period-8, ..., accumulating at a finite ``r`` beyond
which the dynamics are chaotic -- the classic Feigenbaum scenario.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.ode_dynamics.systems.logistic_map import bifurcation_diagram, estimate_feigenbaum_delta
from mathkit.ode_dynamics.visualizers.plots import plot_bifurcation_diagram

# %%
# The bifurcation diagram
# ----------------------------

r_values = np.linspace(2.4, 4.0, 2000)
r_plot, x_plot = bifurcation_diagram(r_values, n_transient=500, n_keep=100)

fig, ax = plt.subplots(figsize=(8, 5))
plot_bifurcation_diagram(r_plot, x_plot, ax=ax)
fig.tight_layout()

# %%
# Estimating the Feigenbaum constant
# ----------------------------------------
# The ratio of successive bifurcation-interval widths converges to the
# universal Feigenbaum constant delta ~ 4.6692.

delta = estimate_feigenbaum_delta()
print(f"estimated Feigenbaum delta (from the first 3 bifurcations): {delta:.4f}")
print("true value: 4.6692...")

plt.show()
