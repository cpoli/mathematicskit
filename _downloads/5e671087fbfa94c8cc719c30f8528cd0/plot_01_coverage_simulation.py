r"""
Confirming a 95% confidence interval's coverage by simulation
====================================================================

Builds many 95% confidence intervals from repeated samples of a known
population, and checks that close to 95% of them actually contain the
true parameter -- the frequentist interpretation of "95% confidence."
"""

# %%
import numpy as np

from mathematicskit.statistics import mean_confidence_interval, proportion_confidence_interval

# %%
# Mean confidence interval coverage
# -----------------------------------------------------

rng = np.random.default_rng(0)
true_mean = 50.0
n_trials = 2000
covered = 0
for _ in range(n_trials):
    sample = rng.normal(loc=true_mean, scale=10.0, size=40)
    result = mean_confidence_interval(sample, sigma=10.0)
    covered += result.lower < true_mean < result.upper

print(f"mean CI empirical coverage: {covered / n_trials:.4f} (target 0.95)")

# %%
# Proportion confidence interval
# -----------------------------------------------------

result = proportion_confidence_interval(successes=42, n=100)
print(f"proportion estimate: {result.estimate:.3f}, 95% CI: ({result.lower:.3f}, {result.upper:.3f})")
