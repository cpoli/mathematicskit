r"""
The hat-check problem: derangements via inclusion-exclusion
==================================================================

``n`` people check their hats and receive one back at random; what's
the probability nobody gets their own hat back? The answer converges
to :math:`1/e` remarkably quickly.
"""

# %%
import math

from mathematicskit.combinatorics import derangement_count

# %%
# Probability of a total derangement, for increasing n
# -----------------------------------------------------

for n in range(1, 11):
    d_n = derangement_count(n)
    probability = d_n / math.factorial(n)
    print(f"n={n:2d}: D_n={d_n:>10}, P(derangement) = {probability:.6f} (1/e = {1 / math.e:.6f})")
