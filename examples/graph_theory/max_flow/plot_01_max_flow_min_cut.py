r"""
Maximum flow and the max-flow min-cut theorem
====================================================

The classic CLRS flow network: finds the maximum flow from source to
sink, and confirms the max-flow min-cut theorem -- the minimum cut's
capacity exactly equals the maximum flow value.
"""

# %%
from mathkit.graph_theory import Graph, max_flow_min_cut

# %%
# Build the flow network
# -----------------------------------------------------

g = Graph(6, directed=True)
g.add_edge(0, 1, 16)
g.add_edge(0, 2, 13)
g.add_edge(1, 2, 10)
g.add_edge(2, 1, 4)
g.add_edge(1, 3, 12)
g.add_edge(2, 4, 14)
g.add_edge(3, 2, 9)
g.add_edge(4, 3, 7)
g.add_edge(3, 5, 20)
g.add_edge(4, 5, 4)

# %%
# Solve and verify the max-flow min-cut theorem
# -----------------------------------------------------

result = max_flow_min_cut(g, source=0, sink=5)
print(f"maximum flow: {result.flow_value}")

source_side, sink_side = result.min_cut
capacity = g.to_sparse().toarray()
cut_capacity = sum(capacity[u, v] for u in source_side for v in sink_side)
print(f"min cut: source side={list(source_side)}, sink side={list(sink_side)}, capacity={cut_capacity}")
