r"""
Cyclic groups vs. the symmetric group
============================================

Compares the abelian cyclic group Z_6 against the non-abelian
symmetric group S_3 -- both have order 6, but very different
structure, visible directly in their Cayley tables.
"""

# %%
from mathkit.abstract_algebra import CyclicGroup, PermutationGroup, group_properties, is_cyclic
from mathkit.abstract_algebra.visualizers.plots import plot_cayley_table

# %%
# Z_6: cyclic, abelian
# -----------------------------------------------------

z6 = CyclicGroup(6)
result_z6 = group_properties(z6)
print(f"Z_6: order={result_z6.order}, abelian={result_z6.is_abelian}, cyclic={is_cyclic(z6)}")
print(f"  element orders: {result_z6.element_orders}")

plot_cayley_table(z6)

# %%
# S_3: non-abelian, not cyclic
# -----------------------------------------------------

s3 = PermutationGroup(3)
result_s3 = group_properties(s3)
print(f"\nS_3: order={result_s3.order}, abelian={result_s3.is_abelian}, cyclic={is_cyclic(s3)}")
print(f"  element orders: {result_s3.element_orders}")

plot_cayley_table(s3)
