Breakthroughs in Graph Theory
=============================


.. include:: /_generated/nav/graph_theory.rst

.. epigraph::

   "This question is so banal, but seemed to me worthy of attention in
   that neither geometry, nor algebra, nor even the art of counting was
   sufficient to solve it."
   -- Leonhard Euler, letter to Giovanni Marinoni, 1736

Graph theory began as a recreational puzzle about a Prussian city's
bridges; nearly three centuries later its algorithms route internet
traffic, plan delivery networks, and cluster social networks. This
chronology traces the ideas behind :mod:`mathematicskit.graph_theory`,
from Leonhard Euler's original impossibility proof to the spectral
methods that read a graph's structure directly from a matrix's
eigenvalues.

.. contents:: Timeline
   :local:
   :depth: 1

1736 -- Euler and the Seven Bridges of Königsberg
-------------------------------------------------

Leonhard Euler's 1736 paper settled a local puzzle in Königsberg (now
Kaliningrad): could a walker cross each of the city's seven bridges
exactly once? Euler proved it impossible by reducing the city to what
would now be called a graph, with land masses as vertices and bridges as
edges. Such a walk requires that at most two land masses touch an odd
number of bridges, and all four of Königsberg's did. Carl Hierholzer
proved in 1873 that the condition is also sufficient for a connected
graph. Euler's paper is universally credited as the founding paper of
graph theory, even though it never draws anything resembling a modern
graph diagram.

*Connection:* every algorithm in :mod:`mathematicskit.graph_theory`
operates on :class:`mathematicskit.graph_theory.core.base.Graph`, the
same vertices-and-edges abstraction Euler introduced to solve this
problem.

*References:* L. Euler, "Solutio problematis ad geometriam situs
pertinentis," Commentarii Academiae Scientiarum Petropolitanae 8 (1741),
128-140 (presented to the St. Petersburg Academy in 1735; the volume is
dated 1736 but was printed in 1741).

.. minigallery:: ../../examples/graph_theory/shortest_paths/plot_01_dijkstra_vs_bellman_ford.py

1852-1976 -- The Four Color Problem
-----------------------------------

In 1852 Francis Guthrie, while coloring a map of England's counties,
conjectured that four colors always suffice to color any planar map so
that no two adjacent regions share a color. Despite its simple
statement, the conjecture resisted proof for a century and a quarter.
Kenneth Appel and Wolfgang Haken's 1976 proof reduced the problem to a
large but finite set of configurations and checked each one by
computer. It was the first major theorem whose proof no person could
feasibly verify by hand, a landmark in how mathematics is done. Coloring
an arbitrary graph (not necessarily planar) with as few colors as
possible is NP-complete, and no efficient exact algorithm is known for
large graphs.

*Implementation:* :func:`mathematicskit.graph_theory.systems.coloring.greedy_coloring`
implements the fast, but not always optimal, greedy heuristic.
:func:`~mathematicskit.graph_theory.systems.coloring.backtracking_coloring`
implements exact exhaustive search. It is feasible only for small
graphs, which is the exponential cost that NP-completeness predicts.

*References:* K. Appel and W. Haken, "Every Planar Map is Four
Colorable," Illinois Journal of Mathematics 21(3) (1977), 429-490
(the published form of the proof announced in 1976).

.. minigallery:: ../../examples/graph_theory/coloring/plot_01_greedy_vs_optimal.py

1927-1956 -- Menger, Ford, Fulkerson, and Max-Flow Min-Cut
----------------------------------------------------------

Karl Menger's 1927 theorem on the number of vertex-disjoint paths
between two vertices anticipated, in a purely combinatorial setting,
the duality at the heart of Lester Ford Jr. and Delbert Fulkerson's
1956 max-flow min-cut theorem. The maximum flow that can be pushed from
a source to a sink through a network with edge capacities equals the
minimum total capacity of any set of edges whose removal disconnects
the source from the sink. Ford and Fulkerson's augmenting-path algorithm
remains the conceptual backbone of modern max-flow solvers. Yefim
Dinitz (published as "Dinic") gave it an efficient blocking-flow form in
1970, and Jack Edmonds and Richard Karp proved a polynomial time bound
for a shortest-augmenting-path version in 1972.

*Implementation:* :func:`mathematicskit.graph_theory.systems.max_flow.max_flow_min_cut`
wraps :func:`scipy.sparse.csgraph.maximum_flow` (Dinitz's algorithm by
default). It derives the corresponding minimum cut from the residual
graph of the resulting flow, which is the certificate the max-flow
min-cut theorem guarantees.

*References:* K. Menger, "Zur allgemeinen Kurventheorie," Fundamenta
Mathematicae 10 (1927), 96-115; L. R. Ford Jr. and D. R. Fulkerson,
"Maximal Flow Through a Network," Canadian Journal of Mathematics 8
(1956), 399-404.

.. minigallery:: ../../examples/graph_theory/max_flow/plot_01_max_flow_min_cut.py

1956-1957 -- Kruskal, Prim, and the Minimum Spanning Tree
---------------------------------------------------------

Joseph Kruskal's 1956 algorithm builds a minimum spanning tree by adding
edges in increasing order of weight, skipping any edge that would close
a cycle. Robert Prim's 1957 algorithm instead grows a single tree
outward, always adding the cheapest edge that leaves the tree so far.
Prim had rediscovered a method that Vojtěch Jarník published in Czech in
1930, which went largely unnoticed outside Central Europe for decades.
Both algorithms are provably optimal and run in similar time, but their
strategies differ -- building a forest versus growing a single tree --
and the difference is easy to see when they run side by side on the
same graph.

*Implementation:* :func:`mathematicskit.graph_theory.systems.spanning_tree.kruskal_mst`
wraps :func:`scipy.sparse.csgraph.minimum_spanning_tree` (Kruskal-based).
:func:`~mathematicskit.graph_theory.systems.spanning_tree.prim_mst` keeps
Prim's algorithm hand-rolled for this side-by-side comparison, not as a
competing primary API.

*References:* J. B. Kruskal, "On the Shortest Spanning Subtree of a
Graph and the Traveling Salesman Problem," Proceedings of the American
Mathematical Society 7(1) (1956), 48-50; R. C. Prim, "Shortest
Connection Networks and Some Generalizations," Bell System Technical
Journal 36(6) (1957), 1389-1401.

.. minigallery:: ../../examples/graph_theory/spanning_tree/plot_01_kruskal_vs_prim.py

1959 -- Dijkstra's Shortest-Path Algorithm
------------------------------------------

Edsger Dijkstra devised his shortest-path algorithm in 1956 while
looking for a simple, useful problem to demonstrate a new computer:
finding the shortest route between two Dutch cities. By his own later
account, he designed it in about twenty minutes without pen or paper.
Published in 1959, the algorithm greedily extends the set of vertices
whose shortest distance is known, one vertex at a time. It remains the
standard algorithm for single-source shortest paths whenever edge
weights are non-negative.

*Implementation:* :func:`mathematicskit.graph_theory.systems.shortest_paths.dijkstra_shortest_paths`
wraps :func:`scipy.sparse.csgraph.dijkstra`. For graphs with negative
edge weights, where Dijkstra's greedy assumption breaks down,
:func:`~mathematicskit.graph_theory.systems.shortest_paths.bellman_ford_shortest_paths`
wraps the Bellman-Ford algorithm (Lester Ford Jr., 1956; Richard
Bellman, 1958) instead.

*References:* E. W. Dijkstra, "A Note on Two Problems in Connexion with
Graphs," Numerische Mathematik 1 (1959), 269-271.

.. minigallery:: ../../examples/graph_theory/shortest_paths/plot_01_dijkstra_vs_bellman_ford.py

1973 -- Fiedler and Spectral Graph Theory
-----------------------------------------

Miroslav Fiedler's 1973 paper studied the eigenvalues of a graph's
Laplacian matrix, :math:`L = D - A` (degree matrix minus adjacency
matrix). Its second-smallest eigenvalue, now called the "algebraic
connectivity" or Fiedler value, is zero exactly when the graph is
disconnected, and otherwise measures how well connected the graph is.
The sign pattern of the corresponding eigenvector splits the vertices
into two well-separated communities, the simplest form of spectral
clustering.

*Implementation:* :func:`mathematicskit.graph_theory.systems.spectral.spectral_analysis`
computes this Laplacian (with :func:`scipy.sparse.csgraph.laplacian`),
its full spectrum (with :func:`numpy.linalg.eigh`), and the
Fiedler-vector bipartition.

*References:* M. Fiedler, "Algebraic Connectivity of Graphs,"
Czechoslovak Mathematical Journal 23(2) (1973), 298-305.

.. minigallery:: ../../examples/graph_theory/spectral/plot_01_barbell_bipartition.py

See Also
--------

- :doc:`/api/graph_theory`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/optimization_breakthroughs`
