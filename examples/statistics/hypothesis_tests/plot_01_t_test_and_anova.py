r"""
Two-sample t-test and one-way ANOVA
==========================================

Compares three fertilizer treatments' effect on plant growth: a
pairwise t-test between two of them, and a one-way ANOVA across all
three, alongside an effect-size measure (Cohen's d).
"""

# %%
import numpy as np

from mathematicskit.statistics import cohens_d, one_way_anova, two_sample_t_test

# %%
# Simulated data: three treatment groups
# -----------------------------------------------------

rng = np.random.default_rng(0)
control = rng.normal(loc=10.0, scale=2.0, size=30)
treatment_a = rng.normal(loc=11.5, scale=2.0, size=30)
treatment_b = rng.normal(loc=13.0, scale=2.0, size=30)

# %%
# Pairwise comparison: control vs. treatment_b
# -----------------------------------------------------

t_result = two_sample_t_test(control, treatment_b)
d = cohens_d(control, treatment_b)
print(f"t={t_result.statistic:.3f}, p={t_result.p_value:.4f}, Cohen's d={d:.3f}")

# %%
# One-way ANOVA across all three groups
# -----------------------------------------------------

anova_result = one_way_anova(control, treatment_a, treatment_b)
print(f"F={anova_result.statistic:.3f}, p={anova_result.p_value:.6f}")
print("group means:", anova_result.extra["group_means"])
