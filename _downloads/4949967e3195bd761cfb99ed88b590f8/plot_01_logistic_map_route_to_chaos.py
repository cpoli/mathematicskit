r"""
Lyapunov exponent across the logistic map's route to chaos
=================================================================

Sweeps the logistic map's growth-rate parameter ``r`` and estimates the
Lyapunov exponent at each value: negative in the periodic windows,
crossing zero exactly at each bifurcation, and mostly positive beyond
the onset of chaos at ``r ~ 3.56995``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.fractals_chaos import lyapunov_exponent_1d_map

# %%
# Sweep r and estimate the exponent at each value
# ------------------------------------------------------

r_values = np.linspace(2.5, 4.0, 400)
exponents = np.empty_like(r_values)
for i, r in enumerate(r_values):
    f = lambda x, r=r: r * x * (1.0 - x)
    fprime = lambda x, r=r: r - 2.0 * r * x
    exponents[i] = lyapunov_exponent_1d_map(f, fprime, x0=0.4, n_transient=500, n_iterations=2000)

# %%
# Plot: exponent crosses zero at each period-doubling bifurcation
# --------------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(r_values, exponents, lw=0.8)
ax.axhline(0.0, color="black", lw=0.6)
ax.set_xlabel("r")
ax.set_ylabel("Lyapunov exponent")
ax.set_title("Logistic map: Lyapunov exponent vs. growth rate")
print("fraction of chaotic (positive-exponent) r values sampled:", float(np.mean(exponents > 0)))
