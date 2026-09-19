mathematicskit.graph_theory
===========================


.. include:: /_generated/nav/graph_theory.rst

A lightweight own graph container (no ``networkx`` dependency);
shortest-path algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall) via
``scipy.sparse.csgraph``; minimum spanning tree (Kruskal via scipy,
Prim hand-rolled for comparison); maximum flow/minimum cut via
``scipy.sparse.csgraph.maximum_flow``; graph coloring (greedy and exact
backtracking, hand-rolled -- NP-complete, no scipy equivalent); and
spectral graph theory (Laplacian via scipy, eigendecomposition via
``numpy.linalg.eigh``).

.. automodule:: mathematicskit.graph_theory
   :members:
   :undoc-members:
