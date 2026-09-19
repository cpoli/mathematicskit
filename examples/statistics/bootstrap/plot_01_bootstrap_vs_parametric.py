r"""
Bootstrap vs. parametric confidence intervals
====================================================

Compares a bootstrap confidence interval for the median (which has no
simple closed form) against the parametric t-based interval for the
mean, on the same skewed dataset.
"""

# %%
import numpy as np

from mathkit.statistics import bootstrap_confidence_interval, mean_confidence_interval

# %%
# A skewed dataset where the mean and median tell different stories
# -------------------------------------------------------------------------

rng = np.random.default_rng(0)
data = rng.exponential(scale=5.0, size=300)

mean_result = mean_confidence_interval(data, sigma=None)
median_result = bootstrap_confidence_interval(data, statistic=np.median, n_resamples=5000, seed=0)

print(f"mean:   {mean_result.estimate:.3f}, 95% CI ({mean_result.lower:.3f}, {mean_result.upper:.3f})")
print(f"median: {median_result.estimate:.3f}, 95% CI ({median_result.lower:.3f}, {median_result.upper:.3f})")
