r"""
Hölder's quotient groups: S_4 / V_4 is S_3
================================================

Finds the normal Klein four-group V_4 inside S_4, forms the quotient
S_4/V_4, and shows that it is a non-abelian group of order 6, like S_3.
"""

# %%
from mathematicskit.abstract_algebra import PermutationGroup, all_subgroups, is_normal_subgroup, quotient_group
from mathematicskit.abstract_algebra.visualizers.plots import plot_cayley_table

# %%
# Normal subgroups of S_4
# -----------------------------------------------------

s4 = PermutationGroup(4)
normal = [h for h in all_subgroups(s4) if is_normal_subgroup(s4, h)]
print(f"normal subgroup orders in S_4: {[len(h) for h in normal]}")

# %%
# The quotient by the Klein four-group
# -----------------------------------------------------

v4 = next(h for h in normal if len(h) == 4)
q = quotient_group(s4, v4)
print(f"|S_4 / V_4| = {q.order}, abelian: {q.is_abelian()}")
print(f"element orders in the quotient: {sorted(q.element_order(c) for c in q.elements)} (same as S_3)")
plot_cayley_table(q)
