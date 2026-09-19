r"""
Lagrange's theorem: subgroup order divides group order
==============================================================

Enumerates every subgroup of Z_12 and confirms each subgroup's order
divides 12, and that its left cosets partition the whole group.
"""

# %%
from mathematicskit.abstract_algebra import CyclicGroup, all_subgroups, cyclic_subgroup, left_cosets

# %%
# Every subgroup of Z_12
# -----------------------------------------------------

g = CyclicGroup(12)
for subgroup in sorted(all_subgroups(g), key=len):
    print(f"subgroup of order {len(subgroup)}: {sorted(subgroup)} (divides 12: {12 % len(subgroup) == 0})")

# %%
# Cosets of the order-3 subgroup {0, 4, 8}
# -----------------------------------------------------

h = cyclic_subgroup(g, 4)
cosets = left_cosets(g, h)
print(f"\nsubgroup H = {h}")
print(f"cosets of H: {cosets}")
print(f"index [G:H] = {len(cosets)} = |G|/|H| = {g.order}/{len(h)} = {g.order // len(h)}")
