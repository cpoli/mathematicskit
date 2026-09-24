r"""
The Weierstrass function: continuous, nowhere differentiable
==================================================================

Plots the Weierstrass function sum a^n cos(b^n pi x) at several zoom
levels. The graph looks equally rough at every scale, difference
quotients grow without bound as the step shrinks, and the graph's
measured length keeps growing as it is sampled more finely.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import weierstrass_function

# %%
# Zooming in never smooths the graph
# -----------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, width in zip(axes, (2.0, 0.2, 0.02)):
    x = np.linspace(0.3 - width / 2, 0.3 + width / 2, 4000)
    ax.plot(x, weierstrass_function(x), lw=0.7)
    ax.set_title(f"window width {width}")
fig.suptitle("W(x) with a = 0.5, b = 7")

# %%
# Difference quotients diverge
# -----------------------------------------------------

for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    slope = (weierstrass_function(0.3 + h) - weierstrass_function(0.3)) / h
    print(f"h = {h:.0e}: difference quotient {float(slope):12.2f}")

# %%
# The graph has infinite length
# -----------------------------------------------------

for n in (10**3, 10**4, 10**5, 10**6):
    x = np.linspace(0, 1, n)
    y = weierstrass_function(x)
    length = np.sum(np.hypot(np.diff(x), np.diff(y)))
    print(f"{n:>8d} samples: polygonal length of the graph over [0, 1] = {length:8.1f}")
