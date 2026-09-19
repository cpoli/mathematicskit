r"""
Integer partitions, the partition function, and Young diagrams
======================================================================

Enumerates every partition of 6, checks the count against the
partition function ``p(6)``, and visualizes one partition's Young
diagram and its conjugate.
"""

# %%
from mathkit.combinatorics import YoungDiagram, integer_partitions, partition_function
from mathkit.combinatorics.visualizers.plots import plot_partition_counts, plot_young_diagram

# %%
# Enumerate the partitions of 6
# -----------------------------------------------------

partitions = integer_partitions(6)
print(f"p(6) = {partition_function(6)}, enumerated {len(partitions)} partitions:")
for p in partitions:
    print(" ", p)

# %%
# A Young diagram and its conjugate
# -----------------------------------------------------

diagram = YoungDiagram([4, 2, 1])
print("\nFerrers diagram of (4, 2, 1):")
print(diagram.ferrers_diagram())
print("conjugate partition:", diagram.conjugate().parts)

plot_young_diagram(diagram)

# %%
# Growth of the partition function
# -----------------------------------------------------

plot_partition_counts(30)
