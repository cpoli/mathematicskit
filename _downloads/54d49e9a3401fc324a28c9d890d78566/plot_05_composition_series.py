r"""
Jordan-Hölder: composition series and their factors
=========================================================

Builds composition series for several groups and lists the orders of
their simple factors. Different groups of the same order can share the
same factors, and the Jordan-Hölder theorem says that every series of
one group has the same factors.
"""

# %%
from mathematicskit.abstract_algebra import CyclicGroup, DihedralGroup, PermutationGroup, composition_series

# %%
# Composition factors
# -----------------------------------------------------

groups = {"Z_12": CyclicGroup(12), "D_6": DihedralGroup(6), "S_4": PermutationGroup(4), "Z_24": CyclicGroup(24)}
for name, group in groups.items():
    result = composition_series(group)
    orders = [len(h) for h in result.series]
    print(f"{name}: series orders {orders}, factor orders {result.factor_orders}")

# %%
# Same factors, different groups
# -----------------------------------------------------
#
# Z_12 (abelian) and D_6 (non-abelian) are not isomorphic, yet both have
# composition factors of orders 2, 2, 3. The Jordan-Hölder theorem makes
# the factors an invariant of a group, but not a complete one.
