r"""
Hamilton's quaternions: the group Q_8
===========================================

Verifies Hamilton's relations i^2 = j^2 = k^2 = ijk = -1 in the
quaternion group, shows that it is non-abelian, and prints its
multiplication table.
"""

# %%
from mathematicskit.abstract_algebra import QuaternionGroup, all_subgroups, is_normal_subgroup
from mathematicskit.abstract_algebra.visualizers.plots import plot_cayley_table

# %%
# Hamilton's relations
# -----------------------------------------------------

q = QuaternionGroup()
for unit in ("i", "j", "k"):
    print(f"{unit}^2 = {q.operate(unit, unit)}")
print(f"ijk = {q.operate(q.operate('i', 'j'), 'k')}")
print(f"ij = {q.operate('i', 'j')}, but ji = {q.operate('j', 'i')}")

# %%
# Multiplication table
# -----------------------------------------------------

print("\n      " + "".join(f"{b:>4}" for b in q.elements))
for a in q.elements:
    print(f"{a:>4}  " + "".join(f"{q.operate(a, b):>4}" for b in q.elements))

# %%
# Non-abelian, yet every subgroup is normal
# -----------------------------------------------------

subgroups = all_subgroups(q)
print(f"\n{len(subgroups)} subgroups, all normal: {all(is_normal_subgroup(q, h) for h in subgroups)}")
plot_cayley_table(q)
