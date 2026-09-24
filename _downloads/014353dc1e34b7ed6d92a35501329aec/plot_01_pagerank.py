r"""
PageRank: ranking pages by a random surfer
================================================

Ranks the vertices of a small web of links by the stationary
distribution of a random surfer who follows links with probability 0.85
and otherwise jumps to a random page, and shows the convergence of the
power iteration.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import Graph, pagerank

# %%
# A small web
# -----------------------------------------------------

links = {0: [1, 2], 1: [2], 2: [0], 3: [0, 2], 4: [0, 3, 5], 5: [4], 6: [2]}
g = Graph(7, directed=True)
for u, targets in links.items():
    for v in targets:
        g.add_edge(u, v)
result = pagerank(g)
for page in np.argsort(result.scores)[::-1]:
    print(f"page {page}: {result.scores[page]:.4f}  (in-links from {[u for u in links if page in links[u]]})")
print(f"converged in {result.iterations} iterations")

# %%
# Convergence at the rate of the damping factor
# -----------------------------------------------------

exact = result.scores
fig, ax = plt.subplots()
for d in (0.5, 0.85, 0.95):
    errors = [np.abs(pagerank(g, damping=d, max_iter=k, tol=0).scores - pagerank(g, damping=d).scores).sum() for k in range(1, 60)]
    ax.semilogy(range(1, 60), errors, label=f"damping {d}")
ax.set_xlabel("iteration")
ax.set_ylabel("L1 error")
ax.legend()
