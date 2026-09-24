r"""
König's theorem and Hopcroft-Karp matching
================================================

Finds a maximum matching in a random bipartite graph with the
Hopcroft-Karp algorithm and builds a vertex cover of exactly the same
size, as König's theorem guarantees.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import Graph, bipartite_matching

# %%
# A random bipartite graph
# -----------------------------------------------------

rng = np.random.default_rng(5)
n_left, n_right = 8, 7
g = Graph(n_left + n_right)
for u in range(n_left):
    for v in rng.choice(n_right, size=rng.integers(1, 3), replace=False):
        g.add_edge(u, n_left + int(v))
result = bipartite_matching(g, left=range(n_left))
print(f"maximum matching size {result.size}: {result.pairs}")
print(f"minimum vertex cover size {len(result.vertex_cover)}: {result.vertex_cover}")

pos = {u: (0, -u) for u in range(n_left)} | {n_left + v: (2, -v - 0.5) for v in range(n_right)}
fig, ax = plt.subplots(figsize=(5, 6))
for u, v, _ in g.edges():
    matched = result.pairs.get(min(u, v)) == max(u, v)
    ax.plot(*zip(pos[u], pos[v]), color="tab:red" if matched else "0.8", lw=3 if matched else 1)
for v, (x, y) in pos.items():
    ax.plot(x, y, "s" if v in result.vertex_cover else "o", color="tab:blue" if v in result.vertex_cover else "k", ms=10)
ax.axis("off")
ax.set_title("matching (red) and vertex cover (blue squares)")
