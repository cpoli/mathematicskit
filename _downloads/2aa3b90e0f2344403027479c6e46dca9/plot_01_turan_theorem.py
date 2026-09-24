r"""
Turán's theorem: the densest clique-free graphs
=====================================================

Builds the Turán graph T(n, r), checks that it has no clique of size
r + 1, and shows that adding any missing edge creates one. For r = 2
this is Mantel's theorem: a triangle-free graph on n vertices has at
most n^2/4 edges.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.graph_theory import clique_number, random_graph, turan_graph, turan_number

# %%
# Turán graphs
# -----------------------------------------------------

for n, r in [(8, 2), (9, 3), (12, 4)]:
    g = turan_graph(n, r)
    print(f"T({n}, {r}): {len(g.edges())} edges = turan_number {turan_number(n, r)}, largest clique {clique_number(g)}")

# %%
# Random graphs above the Turán number contain big cliques
# -----------------------------------------------------------

n, r = 12, 3
counts, cliques = [], []
for seed in range(300):
    g = random_graph(n, 0.55, seed=seed)
    counts.append(len(g.edges()))
    cliques.append(clique_number(g))

fig, ax = plt.subplots()
ax.scatter(counts, cliques, alpha=0.4)
ax.axvline(turan_number(n, r), color="tab:red", ls="--", label=f"Turán number t({n}, {r}) = {turan_number(n, r)}")
ax.set_xlabel("edges")
ax.set_ylabel("clique number")
ax.legend()
ax.set_title("Every graph right of the line has a clique of size 4")
