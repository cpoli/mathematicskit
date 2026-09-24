r"""
Douglas-Peucker line simplification
=========================================

Simplifies a wiggly coastline-like curve at several tolerances. Larger
tolerances keep fewer vertices, and every discarded vertex stays within
the tolerance of the simplified line.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import douglas_peucker

# %%
# A rough curve and its simplifications
# -----------------------------------------------------

rng = np.random.default_rng(2)
x = np.linspace(0, 10, 2000)
y = np.sin(x) + 0.3 * np.sin(5 * x) + 0.05 * np.cumsum(rng.normal(size=x.size)) / np.sqrt(x.size) * 10
line = np.column_stack([x, y])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(*line.T, color="0.7", lw=3, label=f"original ({len(line)} points)")
for eps, color in ((0.05, "tab:blue"), (0.2, "tab:green"), (0.6, "tab:red")):
    simple = douglas_peucker(line, eps)
    ax.plot(*simple.T, "o-", color=color, ms=3, lw=1, label=f"epsilon = {eps}: {len(simple)} points")
ax.legend()
ax.set_title("Douglas-Peucker (1973)")
