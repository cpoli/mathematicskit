r"""
Abel-Ruffini and solvable groups: why S_5 blocks the quintic
==================================================================

Computes the derived series of the symmetric groups S_2 to S_5. It
reaches the trivial group for n <= 4 but stalls at A_5 for n = 5, the
group-theoretic reason the general quintic has no formula in radicals.
"""

# %%
from mathematicskit.abstract_algebra import PermutationGroup, derived_series, is_solvable

# %%
# Derived series of S_n
# -----------------------------------------------------

for n in range(2, 6):
    group = PermutationGroup(n)
    orders = [len(h) for h in derived_series(group)]
    print(f"S_{n}: derived series orders {orders} -> solvable: {is_solvable(group)}")

# %%
# The obstruction
# -----------------------------------------------------
#
# The series for S_5 stops at A_5 (order 60), which equals its own
# commutator subgroup. By Galois's criterion, a degree-5 polynomial whose
# Galois group is S_5 cannot be solved by radicals.
