r"""
Law of Large Numbers and Central Limit Theorem
=====================================================

The sample mean of a Poisson(mu=3) sample converges to its true mean as
the sample size grows (LLN); standardized sample means across many
trials approach a standard normal distribution regardless of the
Poisson's own (skewed, discrete) shape (CLT).
"""

# %%
from scipy import stats

from mathkit.probability import Poisson
from mathkit.probability.systems.limit_theorems import central_limit_theorem_sample_means, law_of_large_numbers_trace
from mathkit.probability.visualizers.plots import plot_clt_histogram

dist = Poisson(mu=3.0)

# %%
# Law of Large Numbers: the sample mean converges to mu=3
# ------------------------------------------------------------------

means = law_of_large_numbers_trace(dist, n_values=[10, 100, 1000, 10000, 1000000], seed=0)
for n, m in zip([10, 100, 1000, 10000, 1000000], means):
    print(f"n={n:>8}: sample mean = {m:.4f}")

# %%
# Central Limit Theorem: standardized sample means approach N(0, 1)
# ------------------------------------------------------------------

z = central_limit_theorem_sample_means(dist, n=200, n_trials=5000, seed=0)
_, p_value = stats.kstest(z, "norm")
print(f"Kolmogorov-Smirnov test against N(0,1): p-value = {p_value:.4f}")

plot_clt_histogram(z)
