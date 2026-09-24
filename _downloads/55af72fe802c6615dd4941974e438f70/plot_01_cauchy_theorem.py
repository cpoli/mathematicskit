r"""
Cauchy's theorem: elements of prime order
===============================================

For each prime p dividing a group's order, finds the elements of order
p, and checks McKay's refinement that their number is congruent to -1
modulo p.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.abstract_algebra import DihedralGroup, PermutationGroup, QuaternionGroup, elements_of_order

# %%
# Elements of each prime order
# -----------------------------------------------------

groups = {"S_4": PermutationGroup(4), "D_5": DihedralGroup(5), "Q_8": QuaternionGroup(), "S_5": PermutationGroup(5)}
counts = {}
for name, group in groups.items():
    for p in (2, 3, 5):
        if group.order % p == 0:
            count = len(elements_of_order(group, p))
            counts[(name, p)] = count
            print(f"{name} (order {group.order}): {count} elements of order {p}, count mod {p} = {count % p}")

# %%
# Every count is -1 mod p
# -----------------------------------------------------

labels = [f"{name}, p={p}" for name, p in counts]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(labels, list(counts.values()), color="tab:blue")
ax.set_ylabel("elements of order p")
ax.set_title("Cauchy's theorem: every prime divisor p of |G| has elements of order p")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
