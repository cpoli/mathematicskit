r"""
Bellman's principle of optimality: the 0/1 knapsack
=========================================================

Fills a table of optimal values for every prefix of the items and every
capacity, then traces back the chosen items. The table is the
dynamic-programming "value function".
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.optimization import knapsack

# %%
# Six items, capacity 15
# -----------------------------------------------------

values = [10, 13, 7, 8, 15, 4]
weights = [5, 6, 3, 4, 7, 2]
capacity = 15
result = knapsack(values, weights, capacity)
print(f"best value {result.value:.0f} using items {result.items.tolist()} (total weight {result.weight})")

# %%
# The value table V(i, c)
# -----------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(result.table, cmap="viridis", aspect="auto", origin="lower")
fig.colorbar(im, ax=ax, label="best value")
ax.set_xlabel("capacity c")
ax.set_ylabel("items considered i")
ax.set_title("Bellman's value table for the knapsack problem (1957)")
