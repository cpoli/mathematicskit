Breakthroughs in Graph Theory
================================


.. include:: /_generated/nav/graph_theory.rst

.. epigraph::

   "A graph is a mathematical structure used to model pairwise relations
   between objects" -- and, as this chronology traces, one that keeps
   turning out to model an implausibly wide range of them.

Graph theory's founding problem was a recreational puzzle about a
Prussian city's bridges; two and a half centuries later its algorithms
route internet traffic, plan delivery networks, and cluster social
networks. This chronology traces the ideas behind
:mod:`mathematicskit.graph_theory`, from Euler's original impossibility proof
to the spectral methods that read a graph's structure directly out of a
matrix's eigenvalues.

.. contents:: Timeline
   :local:
   :depth: 1

1736 -- Euler and the Seven Bridges of Konigsberg
-------------------------------------------------------

Leonhard Euler's 1736 paper answered a standing local puzzle in
Konigsberg (now Kaliningrad): could a walker cross all seven of the
city's bridges exactly once each and return to their starting point?
Euler proved it impossible by abstracting the city to what would now be
called a graph -- landmasses as vertices, bridges as edges -- and showing
that such a closed walk (now called an Eulerian circuit) exists if and
only if every vertex has even degree, which Konigsberg's bridge layout
did not satisfy. It is universally credited as the founding paper of
graph theory, even though Euler's own paper never draws anything that
looks like a modern graph diagram.

*Connection:* every algorithm in :mod:`mathematicskit.graph_theory` operates on
:class:`mathematicskit.graph_theory.core.base.Graph`, the same
vertices-and-edges abstraction Euler introduced to solve this problem.

*References:* L. Euler, "Solutio problematis ad geometriam situs
pertinentis," Commentarii Academiae Scientiarum Petropolitanae 8 (1741),
128-140 (presented to the St. Petersburg Academy in 1735, published
1741).

1959 -- Dijkstra's Shortest-Path Algorithm
------------------------------------------------

Edsger Dijkstra devised his shortest-path algorithm in 1956, by his own
later account in about twenty minutes without pen or paper, while
thinking through how to demonstrate a new computer's capabilities with a
simple, useful problem: find the shortest route between two Dutch
cities. Published in 1959, the algorithm greedily extends the
shortest-known-distance frontier one vertex at a time, and remains the
standard algorithm for single-source shortest paths whenever edge
weights are non-negative.

*Implementation:* :func:`mathematicskit.graph_theory.systems.shortest_paths.dijkstra_shortest_paths`
wraps :func:`scipy.sparse.csgraph.dijkstra`; for graphs with negative
edge weights (where Dijkstra's greedy assumption breaks down),
:func:`~mathematicskit.graph_theory.systems.shortest_paths.bellman_ford_shortest_paths`
wraps the Bellman-Ford algorithm (Richard Bellman, 1958; Lester Ford
Jr., 1956) instead.

*References:* E. W. Dijkstra, "A Note on Two Problems in Connexion with
Graphs," Numerische Mathematik 1 (1959), 269-271.

.. minigallery:: ../../examples/graph_theory/shortest_paths/plot_01_dijkstra_vs_bellman_ford.py

1956 -- 1957 -- Kruskal, Prim, and the Minimum Spanning Tree
------------------------------------------------------------------

Joseph Kruskal's 1956 algorithm builds a minimum spanning tree by adding
edges in increasing order of weight, skipping any edge that would close
a cycle among the edges already chosen; Robert Prim's 1957 algorithm
(rediscovering a method Vojtech Jarnik had already published in 1930,
in Czech and largely unnoticed outside Central Europe until decades
later) instead grows a single tree outward, always adding the cheapest
edge leaving the tree so far. Both are provably optimal and run in
similar time, but via genuinely different strategies -- forest-building
versus tree-growing -- a difference visible directly by running them side
by side on the same graph.

*Implementation:* :func:`mathematicskit.graph_theory.systems.spanning_tree.kruskal_mst`
wraps :func:`scipy.sparse.csgraph.minimum_spanning_tree` (Kruskal-based);
:func:`~mathematicskit.graph_theory.systems.spanning_tree.prim_mst` keeps
Prim's algorithm hand-rolled specifically for this side-by-side
comparison, not as a competing primary API.

*References:* J. B. Kruskal, "On the Shortest Spanning Subtree of a
Graph and the Traveling Salesman Problem," Proceedings of the American
Mathematical Society 7(1) (1956), 48-50; R. C. Prim, "Shortest
Connection Networks and Some Generalizations," Bell System Technical
Journal 36(6) (1957), 1389-1401.

.. minigallery:: ../../examples/graph_theory/spanning_tree/plot_01_kruskal_vs_prim.py

1927 -- 1956 -- Menger, Ford, Fulkerson, and Max-Flow Min-Cut
--------------------------------------------------------------------

Karl Menger's 1927 theorem on the number of vertex-disjoint paths
between two points in a graph anticipated, in a purely combinatorial
setting, the duality that Lester Ford Jr. and Delbert Fulkerson's 1956
max-flow min-cut theorem made central to network flow theory: the
maximum amount of flow that can be pushed from a source to a sink
through a capacitated network exactly equals the minimum total capacity
of any set of edges whose removal disconnects the source from the sink.
Ford and Fulkerson's own augmenting-path algorithm, refined for
guaranteed polynomial time by Jack Edmonds and Richard Karp in 1972 and
given its efficient blocking-flow implementation by Yefim Dinic in 1970,
remains the conceptual backbone of every modern max-flow solver.

*Implementation:* :func:`mathematicskit.graph_theory.systems.max_flow.max_flow_min_cut`
wraps :func:`scipy.sparse.csgraph.maximum_flow` (Dinic's algorithm by
default), and derives the corresponding minimum cut directly from the
resulting flow's residual graph, exactly the certificate the max-flow
min-cut theorem guarantees exists.

*References:* L. R. Ford Jr. and D. R. Fulkerson, "Maximal Flow Through
a Network," Canadian Journal of Mathematics 8 (1956), 399-404.

.. minigallery:: ../../examples/graph_theory/max_flow/plot_01_max_flow_min_cut.py

1852 -- 1976 -- The Four Color Problem
--------------------------------------------

Francis Guthrie, coloring a map of England's counties, conjectured in
1852 that four colors always suffice to color any planar map so that no
two adjacent regions share a color -- a statement of stubborn,
century-and-a-quarter-long difficulty despite its simple appearance.
Kenneth Appel and Wolfgang Haken's 1976 proof, reducing the general
problem to a large but finite set of unavoidable configurations checked
exhaustively by computer, was the first major theorem in mathematics
whose proof no human being could feasibly verify by hand -- itself a
landmark event in how mathematics is done, independent of the specific
result. The general graph-coloring problem (using as few colors as
possible for an arbitrary, not necessarily planar, graph) is NP-complete,
with no known efficient exact algorithm for large graphs.

*Implementation:* :func:`mathematicskit.graph_theory.systems.coloring.greedy_coloring`
implements the fast (but not always optimal) greedy heuristic;
:func:`~mathematicskit.graph_theory.systems.coloring.backtracking_coloring`
implements exact exhaustive search, feasible only for small graphs,
exactly the exponential cost the problem's NP-completeness predicts.

*References:* K. Appel and W. Haken, "Every Planar Map is Four
Colorable," Illinois Journal of Mathematics 21(3) (1977), 429-490
(announcing the 1976 proof).

.. minigallery:: ../../examples/graph_theory/coloring/plot_01_greedy_vs_optimal.py

1973 -- Fiedler and Spectral Graph Theory
-----------------------------------------------

Miroslav Fiedler's 1973 paper studied the eigenvalues of a graph's
Laplacian matrix (:math:`L = D - A`, degree matrix minus adjacency
matrix) and found that its second-smallest eigenvalue -- now called the
"algebraic connectivity" or the Fiedler value in his honor -- is exactly
zero if and only if the graph is disconnected, and otherwise quantifies
how well-connected it is; the corresponding eigenvector's sign pattern
gives a natural way to split a graph's vertices into two well-separated
communities, the simplest form of spectral clustering.

*Implementation:* :func:`mathematicskit.graph_theory.systems.spectral.spectral_analysis`
computes exactly this Laplacian (via
:func:`scipy.sparse.csgraph.laplacian`), its full spectrum (via
:func:`numpy.linalg.eigh`), and the Fiedler-vector bipartition.

*References:* M. Fiedler, "Algebraic Connectivity of Graphs,"
Czechoslovak Mathematical Journal 23(2) (1973), 298-305.

.. minigallery:: ../../examples/graph_theory/spectral/plot_01_barbell_bipartition.py

See Also
--------

- :doc:`/api/graph_theory`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/optimization_breakthroughs`
