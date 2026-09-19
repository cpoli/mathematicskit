r"""
Greedy vs. optimal graph coloring
=======================================

On an odd cycle, the greedy heuristic (with an unlucky vertex order)
can use more colors than the graph's true chromatic number, found by
exact backtracking search.
"""

# %%
from mathematicskit.graph_theory import backtracking_coloring, greedy_coloring
from mathematicskit.graph_theory.utils.generators import cycle_graph
from mathematicskit.graph_theory.visualizers.plots import plot_graph

# %%
# A 5-cycle: chromatic number 3 (odd cycles are never bipartite)
# -------------------------------------------------------------------

g = cycle_graph(5)
greedy_result = greedy_coloring(g)
optimal_result = backtracking_coloring(g)

print(f"greedy:  {greedy_result.num_colors} colors, coloring = {greedy_result.coloring}")
print(f"optimal: {optimal_result.num_colors} colors, coloring = {optimal_result.coloring}")

# %%
# Visualize the optimal coloring
# -----------------------------------------------------

plot_graph(g, vertex_colors=optimal_result.coloring)
