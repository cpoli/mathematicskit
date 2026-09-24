r"""
Hall's marriage theorem
=============================

Checks Hall's condition on random bipartite graphs and confirms that it
holds exactly when a matching covers every left vertex. When it fails,
the example prints the subset whose neighbourhood is too small.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import hall_condition

# %%
# A graph that fails Hall's condition
# -----------------------------------------------------

applicants = {"Ann": ["math"], "Bo": ["math"], "Cy": ["math", "art"], "Di": ["art", "music"]}
result = hall_condition(applicants)
print(f"satisfied: {result.satisfied}, violating subset: {sorted(result.violating_subset)}")
print(f"best matching: {result.matching}")

# %%
# Hall's condition vs. perfect matchings on random graphs
# -----------------------------------------------------------

rng = np.random.default_rng(0)
probabilities = np.linspace(0.1, 0.7, 13)
rates = []
for p in probabilities:
    hits = 0
    for _ in range(200):
        adjacency = {u: [v for v in range(6) if rng.random() < p] for u in range(6)}
        result = hall_condition(adjacency)
        assert result.satisfied == (len(result.matching) == 6)
        hits += result.satisfied
    rates.append(hits / 200)

fig, ax = plt.subplots()
ax.plot(probabilities, rates, "o-")
ax.set_xlabel("edge probability")
ax.set_ylabel("fraction with a perfect matching")
ax.set_title("6 x 6 random bipartite graphs")
