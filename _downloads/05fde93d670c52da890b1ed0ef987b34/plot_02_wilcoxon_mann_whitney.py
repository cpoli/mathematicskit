r"""
Wilcoxon and Mann-Whitney rank tests
=========================================

Rank tests throw away the raw values and keep only their order, so a
few extreme observations cannot dominate. With heavy-tailed data they
detect a shift that the t-test misses.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.statistics import mann_whitney_u_test, two_sample_t_test, wilcoxon_signed_rank_test

# %%
# Two independent samples with heavy tails
# -----------------------------------------------------

rng = np.random.default_rng(1)
a = rng.standard_cauchy(size=60)
b = rng.standard_cauchy(size=60) + 1.0
mw = mann_whitney_u_test(a, b)
tt = two_sample_t_test(a, b, equal_var=False)
print(f"Mann-Whitney: U = {mw.statistic:.0f} of {a.size * b.size}, p = {mw.p_value:.4f}")
print(f"Welch t-test: t = {tt.statistic:.3f}, p = {tt.p_value:.4f}")

# %%
# Paired measurements
# -----------------------------------------------------

before = rng.normal(loc=120.0, scale=10.0, size=15)
after = before - 4.0 + rng.standard_t(df=2, size=15) * 3.0
paired = wilcoxon_signed_rank_test(before, after)
print(f"Wilcoxon signed-rank on before - after: p = {paired.p_value:.4f}")

fig, ax = plt.subplots()
ax.boxplot([a, b])
ax.set_xticks([1, 2], ["sample a", "sample b (shifted by 1)"])
ax.set_ylim(-10, 10)
ax.set_title("Cauchy samples, clipped to [-10, 10]")
