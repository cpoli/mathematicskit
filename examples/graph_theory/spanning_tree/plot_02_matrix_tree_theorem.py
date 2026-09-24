r"""
Kirchhoff's matrix-tree theorem
=====================================

Counts spanning trees with a single determinant of the reduced
Laplacian, and checks the count by brute-force enumeration of edge
subsets on a small graph.
"""

# %%
from itertools import combinations

import numpy as np

from mathematicskit.graph_theory import Graph, complete_bipartite_graph, complete_graph, count_spanning_trees, cycle_graph, dodecahedron_graph

# %%
# Determinant against brute force
# -----------------------------------------------------

g = Graph(5)
for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 4), (3, 4)]:
    g.add_edge(u, v)
edges = [(u, v) for u, v, _ in g.edges()]


def is_spanning_tree(subset):
    parent = list(range(5))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for u, v in subset:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True


brute = sum(is_spanning_tree(s) for s in combinations(edges, 4))
print(f"brute force: {brute} spanning trees; matrix-tree theorem: {count_spanning_trees(g)}")

# %%
# Classical counts
# -----------------------------------------------------

print(f"K_6:          {count_spanning_trees(complete_graph(6))} = 6^4 (Cayley)")
print(f"cycle C_9:    {count_spanning_trees(cycle_graph(9))} (drop any one edge)")
print(f"K_(3,4):      {count_spanning_trees(complete_bipartite_graph(3, 4))} = 3^3 * 4^2")
print(f"dodecahedron: {count_spanning_trees(dodecahedron_graph())}")
adjacency = complete_graph(6).to_sparse().toarray()
print("\nreduced Laplacian of K_6:\n", (np.diag(adjacency.sum(1)) - adjacency)[1:, 1:])
