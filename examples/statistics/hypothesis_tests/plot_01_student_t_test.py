r"""
Student's t-distribution and the small-sample t-test
=====================================================

With only a handful of measurements, the standardized mean
:math:`t = (\bar x - \mu_0)/(s/\sqrt n)` is not normal: estimating the
spread from the same small sample gives it heavier tails. Gosset
("Student", 1908) derived its exact distribution. This example shows
the heavy tails, how a normal cutoff misleads for small samples, and
the one- and two-sample t-tests built on Student's distribution.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mathematicskit.statistics import cohens_d, one_sample_t_test, two_sample_t_test

# %%
# Heavier tails than the normal
# -----------------------------------------------------

x = np.linspace(-5.0, 5.0, 400)
fig, ax = plt.subplots()
ax.plot(x, stats.norm.pdf(x), "k--", label="standard normal")
for df in (1, 3, 10):
    ax.plot(x, stats.t.pdf(x, df), label=f"Student t, {df} df")
ax.set_xlabel("t")
ax.set_ylabel("density")
ax.set_yscale("log")
ax.set_ylim(1e-4, 0.5)
ax.legend()

# %%
# Why the normal cutoff fails for small samples
# -----------------------------------------------------
#
# Draw many samples of size 5 from a population whose mean really is
# :math:`\mu_0`. Rejecting when :math:`|t| > 1.96` (the normal 5% cutoff)
# rejects far too often; Student's cutoff restores the 5% rate.

rng = np.random.default_rng(0)
n = 5
samples = rng.normal(loc=10.0, scale=2.0, size=(20000, n))
t_stats = (samples.mean(axis=1) - 10.0) / (samples.std(axis=1, ddof=1) / np.sqrt(n))
t_cutoff = stats.t.ppf(0.975, n - 1)
print(f"normal cutoff 1.96:      false-rejection rate {np.mean(np.abs(t_stats) > 1.96):.3f}")
print(f"Student cutoff {t_cutoff:.3f}: false-rejection rate {np.mean(np.abs(t_stats) > t_cutoff):.3f}")

# %%
# Gosset's setting: a small batch of brewing measurements
# --------------------------------------------------------

batch = np.array([10.8, 11.6, 10.2, 12.1, 11.4, 10.9])
one = one_sample_t_test(batch, mu0=10.0)
print(f"one-sample t = {one.statistic:.3f} on {one.df:.0f} df, p = {one.p_value:.4f}")

old_recipe = rng.normal(loc=10.0, scale=0.8, size=8)
new_recipe = rng.normal(loc=11.0, scale=0.8, size=8)
two = two_sample_t_test(old_recipe, new_recipe)
print(f"two-sample t = {two.statistic:.3f}, p = {two.p_value:.4f}, Cohen's d = {cohens_d(old_recipe, new_recipe):.3f}")
