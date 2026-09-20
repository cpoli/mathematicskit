r"""
Chi-square goodness-of-fit and independence tests
==========================================================

Pearson's chi-square statistic applied two ways: testing whether a
die's observed roll counts match a fair-die model (goodness-of-fit),
and testing whether two categorical variables in a contingency table
are independent.
"""

# %%
import numpy as np

from mathematicskit.statistics import chi_square_goodness_of_fit, chi_square_independence

# %%
# Goodness-of-fit: is this die fair?
# -----------------------------------------------------

rng = np.random.default_rng(0)
rolls = rng.choice(6, size=600, p=[0.14, 0.16, 0.15, 0.15, 0.17, 0.23])
observed = np.array([np.sum(rolls == face) for face in range(6)], dtype=float)
expected = np.full(6, 100.0)

gof_result = chi_square_goodness_of_fit(observed, expected)
print(f"chi2={gof_result.statistic:.3f}, df={gof_result.df:.0f}, p={gof_result.p_value:.4f}")
print("reject fair-die null at alpha=0.05:", gof_result.reject_null(alpha=0.05))

# %%
# Independence: does treatment outcome depend on treatment group?
# -----------------------------------------------------------------------

contingency_table = np.array([[30.0, 10.0], [20.0, 40.0]])
independence_result = chi_square_independence(contingency_table)
print(f"chi2={independence_result.statistic:.3f}, df={independence_result.df:.0f}, p={independence_result.p_value:.6f}")
print("expected counts under independence:\n", independence_result.extra["expected"])
