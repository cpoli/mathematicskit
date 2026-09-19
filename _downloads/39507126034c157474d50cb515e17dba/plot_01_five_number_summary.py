r"""
Descriptive statistics and the five-number summary
=========================================================

Summarizes a skewed dataset (exponential-ish waiting times) and
visualizes its five-number summary as a boxplot.
"""

# %%
import numpy as np

from mathkit.statistics import descriptive_stats
from mathkit.statistics.visualizers.plots import plot_boxplot

# %%
# Generate a right-skewed dataset and summarize it
# -----------------------------------------------------

rng = np.random.default_rng(0)
data = rng.exponential(scale=3.0, size=500)
result = descriptive_stats(data)

print(f"n={result.n}, mean={result.mean:.3f}, std={result.std:.3f}")
print(f"skewness={result.skewness:.3f} (positive: right-skewed, as expected for an exponential)")
print(f"five-number summary: [{result.minimum:.2f}, {result.q1:.2f}, {result.median:.2f}, {result.q3:.2f}, {result.maximum:.2f}]")

# %%
# Visualize the five-number summary
# -----------------------------------------------------

plot_boxplot(data, label="waiting times")
