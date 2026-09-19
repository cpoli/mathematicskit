Breakthroughs in Computational Geometry
==========================================


.. include:: /_generated/nav/geometry.rst

.. epigraph::

   "There is no royal road to geometry." -- Euclid, reportedly to
   Ptolemy I, as recorded by Proclus in his 5th-century commentary on
   the *Elements*

Geometry is arguably mathematics's oldest rigorously axiomatized subject,
yet *computational* geometry -- how to compute a convex hull, a
triangulation, or a curve's curvature efficiently and correctly in
finite arithmetic -- is barely fifty years old as a named discipline.
This chronology traces the ideas behind :mod:`mathematicskit.geometry`, from
Euclid's original axioms to the algorithms that now underlie every
computer-graphics rendering pipeline and geographic information system.

.. contents:: Timeline
   :local:
   :depth: 1

c. 300 BCE -- Euclid's Elements
-------------------------------------

Euclid's *Elements* organized geometry as a deductive system built from
a small set of postulates and common notions, deriving hundreds of
theorems purely by logical inference -- the founding template for
axiomatic mathematics as a whole, not just geometry. Book I's
constructions (with an idealized compass and straightedge) are
themselves a kind of algorithm, and the shoelace-formula-style area
computations this domain relies on descend directly from Book VI's
theory of areas of plane figures.

*Implementation:* :func:`mathematicskit.geometry.systems.polygon.polygon_area`
and :func:`~mathematicskit.geometry.systems.polygon.polygon_centroid` compute
exactly the areas and centroids Euclidean geometry studies, via the
shoelace formula.

*References:* Euclid, *Elements* (c. 300 BCE), Books I and VI, as
translated in T. L. Heath, *The Thirteen Books of Euclid's Elements*,
2nd ed. (Cambridge: Cambridge University Press, 1926).

.. minigallery:: ../../examples/geometry/polygon/plot_01_shoelace_formula.py

1850s -- Frenet, Serret, and the Moving Frame
----------------------------------------------------

Jean Frederic Frenet's 1847 thesis and Joseph Alfred Serret's 1851 paper
independently derived the equations governing a moving orthonormal frame
(tangent, normal, and binormal vectors) attached to a curve in space, and
the two scalar quantities -- curvature and torsion -- that completely
determine the curve's shape up to rigid motion: curvature measures how
sharply the curve bends within its own osculating plane, and torsion
measures how quickly that plane itself twists out of a flat curve.

.. math::

   T' = \kappa N, \qquad N' = -\kappa T + \tau B, \qquad B' = -\tau N

*Implementation:* :func:`mathematicskit.geometry.systems.curves.frenet_serret_frame`
implements exactly this frame and both scalar invariants, computing
derivatives via :func:`numpy.gradient` and arc length via
:func:`scipy.integrate.cumulative_trapezoid`, cross-checked against a
circular helix's closed-form constant curvature and torsion.

*References:* J. F. Frenet, "Sur les courbes a double courbure" (Ph.D.
thesis, Universite de Toulouse, 1847), condensed in Journal de
Mathematiques Pures et Appliquees 17 (1852), 437-447; J. A. Serret, "Sur
quelques formules relatives a la theorie des courbes a double
courbure," Journal de Mathematiques Pures et Appliquees 16 (1851),
193-207.

.. minigallery:: ../../examples/geometry/curves/plot_01_helix_frenet_frame.py

1972 -- Graham's Scan and the Convex Hull
------------------------------------------------

Ronald Graham's 1972 paper, motivated by a statistical problem about the
smallest convex polygon containing a set of sample points, gave one of
the first genuinely efficient convex-hull algorithms: sort the points by
polar angle around a fixed pivot, then sweep them in order, discarding
any point that would make the hull-in-progress turn clockwise rather
than counterclockwise. It runs in :math:`O(n\log n)` time -- dominated
entirely by the initial sort -- and remains one of the clearest
illustrations of how a global geometric structure (the hull) can be
built from a simple, local, per-point decision rule.

*Implementation:* :func:`mathematicskit.geometry.systems.convex_hull.graham_scan`
implements exactly this angular sweep, kept as a pedagogical comparison
against the primary API,
:func:`~mathematicskit.geometry.systems.convex_hull.convex_hull`, which wraps
:class:`scipy.spatial.ConvexHull` (the Qhull library).

*References:* R. L. Graham, "An Efficient Algorithm for Determining the
Convex Hull of a Finite Planar Set," Information Processing Letters
1(4) (1972), 132-133.

.. minigallery:: ../../examples/geometry/convex_hull/plot_01_qhull_vs_graham_scan.py

1934 -- Delaunay Triangulation
-------------------------------------

Boris Delaunay's 1934 paper, building on Georgy Voronoi's 1908 diagrams
(below), defined the triangulation of a point set that maximizes the
minimum angle among all possible triangulations -- equivalently, one in
which no point lies inside any triangle's circumcircle. Delaunay
triangulations avoid the thin, needle-like triangles that make other
triangulations numerically troublesome for interpolation and finite-
element meshing, and are, point for point, exactly the graph dual of the
corresponding Voronoi diagram.

*Implementation:* :func:`mathematicskit.geometry.systems.triangulation.delaunay_triangulation`
wraps :class:`scipy.spatial.Delaunay` (Qhull), and
:func:`~mathematicskit.geometry.systems.triangulation.voronoi_diagram` wraps
:class:`scipy.spatial.Voronoi` for its dual.

*References:* B. Delaunay, "Sur la sphere vide," Bulletin de l'Academie
des Sciences de l'URSS, Classe des Sciences Mathematiques et Naturelles
6 (1934), 793-800.

.. minigallery:: ../../examples/geometry/triangulation/plot_01_delaunay_and_voronoi.py

1908 -- Voronoi Diagrams
--------------------------------

Georgy Voronoi's 1908 paper (generalizing an idea Descartes had sketched
qualitatively in 1644 to describe how the influence of stars might
partition space, and Dirichlet had studied rigorously in two and three
dimensions in 1850) partitioned the plane into regions, one per input
point, consisting of every location closer to that point than to any
other -- a construction now ubiquitous well beyond mathematics, from
modeling crystal grain boundaries to drawing service-area maps for a
network of facilities.

*Implementation:* :func:`mathematicskit.geometry.systems.triangulation.voronoi_diagram`
implements exactly this partition via :class:`scipy.spatial.Voronoi`.

*References:* G. Voronoi, "Nouvelles applications des parametres
continus a la theorie des formes quadratiques," Journal fur die Reine
und Angewandte Mathematik 133 (1908), 97-178.

.. minigallery:: ../../examples/geometry/triangulation/plot_01_delaunay_and_voronoi.py

1962 -- 1974 -- Point-in-Polygon and the Jordan Curve Theorem
-------------------------------------------------------------------

Camille Jordan's 1887 theorem -- that any simple closed curve in the
plane divides it into exactly an inside and an outside region -- sounds
obvious but resisted a fully rigorous proof for decades (Jordan's own
1887 proof was later found to have gaps, only closed satisfactorily by
Oswald Veblen in 1905). The ray-casting algorithm used to *test*
whether a point is inside a polygon computationally -- count how many
times a ray from the point crosses the polygon's boundary, with an odd
count meaning inside -- is a direct, algorithmic corollary of Jordan's
theorem, put into essentially its modern efficient form in the
computational-geometry literature of the early 1970s.

*Implementation:* :func:`mathematicskit.geometry.systems.intersections.point_in_polygon`
implements exactly this ray-casting test, verified directly against a
concave ("L"-shaped) polygon in this domain's tests;
:func:`~mathematicskit.geometry.systems.intersections.segment_intersection`
implements the companion segment-intersection primitive via Cramer's
rule on the two segments' parametric equations.

*References:* C. Jordan, *Cours d'analyse de l'Ecole Polytechnique*,
vol. 3 (Paris: Gauthier-Villars, 1887); O. Veblen, "Theory on Plane
Curves in Non-Metrical Analysis Situs," Transactions of the American
Mathematical Society 6(1) (1905), 83-98.

.. minigallery:: ../../examples/geometry/intersections/plot_01_segments_and_polygons.py

See Also
--------

- :doc:`/api/geometry`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/fractals_chaos_breakthroughs`
