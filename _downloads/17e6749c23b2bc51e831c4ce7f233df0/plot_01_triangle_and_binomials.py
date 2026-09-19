r"""
Pascal's triangle vs. scipy-computed binomial coefficients
==================================================================

Builds Pascal's triangle by hand via the addition recurrence and
confirms every entry matches ``scipy.special.comb``.
"""

# %%
from mathkit.combinatorics import combinations_count, pascals_triangle
from mathkit.combinatorics.visualizers.plots import plot_pascals_triangle

# %%
# Build and cross-check
# -----------------------------------------------------

triangle = pascals_triangle(10)
for n, row in enumerate(triangle):
    print(f"n={n}: {row}")
    for k, value in enumerate(row):
        assert value == combinations_count(n, k)
print("\nevery entry matches scipy.special.comb")

# %%
# Visualize
# -----------------------------------------------------

plot_pascals_triangle(triangle)
