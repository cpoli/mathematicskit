Breakthroughs in Computational Geometry
=======================================


.. include:: /_generated/nav/geometry.rst

.. epigraph::

   "There is no royal road to geometry."
   -- Euclid, reportedly to Ptolemy I, as recorded by Proclus in his
   5th-century commentary on the *Elements*

Geometry is arguably the oldest rigorously axiomatized subject in
mathematics, yet *computational* geometry -- how to compute a convex
hull, a triangulation, or a curve's curvature efficiently and correctly
in finite-precision arithmetic -- has existed as a named discipline for
only about fifty years. This chronology traces the ideas behind
:mod:`mathematicskit.geometry`, from Euclid's axioms to the algorithms
that now underlie computer-graphics pipelines and geographic
information systems.

.. contents:: Timeline
   :local:
   :depth: 1

c. 300 BCE -- Euclid's Elements
-------------------------------

Euclid's *Elements* organized geometry as a deductive system. From a
small set of postulates and common notions it derived hundreds of
theorems by logical inference alone, setting the template for axiomatic
mathematics as a whole. The compass-and-straightedge constructions of
Book I are themselves a kind of algorithm, and the book's theory of
area (Propositions 35-45) is the ancestor of the polygon-area
computations in this domain. The shoelace formula those computations
use came much later: Albrecht Ludwig Friedrich Meister gave it in 1769,
and Carl Friedrich Gauss in 1795.

*Implementation:* :func:`mathematicskit.geometry.systems.polygon.polygon_area`
and :func:`~mathematicskit.geometry.systems.polygon.polygon_centroid`
compute a polygon's area and centroid with the shoelace formula.

*References:* Euclid, *Elements* (c. 300 BCE), Book I, as translated in
T. L. Heath, *The Thirteen Books of Euclid's Elements*, 2nd ed.
(Cambridge: Cambridge University Press, 1926).

.. minigallery:: ../../examples/geometry/polygon/plot_01_shoelace_formula.py

1847-1851 -- Frenet, Serret, and the Moving Frame
-------------------------------------------------

Jean Frédéric Frenet's 1847 thesis and Joseph Alfred Serret's 1851 paper
independently derived the equations for a moving orthonormal frame
(tangent, normal, and binormal vectors) attached to a curve in space.
They also identified the two scalar quantities that determine the
curve's shape up to rigid motion. Curvature measures how sharply the
curve bends within its osculating plane; torsion measures how quickly
that plane twists, pulling the curve out of a flat shape.

.. math::

   T' = \kappa N, \qquad N' = -\kappa T + \tau B, \qquad B' = -\tau N

*Implementation:* :func:`mathematicskit.geometry.systems.curves.frenet_serret_frame`
computes this frame and both scalar invariants. It takes derivatives
with :func:`numpy.gradient` and arc length with
:func:`scipy.integrate.cumulative_trapezoid`, and is checked against the
closed-form constant curvature and torsion of a circular helix.

*References:* J. F. Frenet, "Sur les courbes à double courbure" (Ph.D.
thesis, Université de Toulouse, 1847), condensed in Journal de
Mathématiques Pures et Appliquées 17 (1852), 437-447; J. A. Serret,
"Sur quelques formules relatives à la théorie des courbes à double
courbure," Journal de Mathématiques Pures et Appliquées 16 (1851),
193-207.

.. minigallery:: ../../examples/geometry/curves/plot_01_helix_frenet_frame.py

1887-1962 -- Point-in-Polygon and the Jordan Curve Theorem
----------------------------------------------------------

Camille Jordan's 1887 theorem states that any simple closed curve in
the plane divides it into exactly two regions, an inside and an
outside. It sounds obvious but is hard to prove. Jordan's own proof was
long considered incomplete, and Oswald Veblen's 1905 proof is usually
credited as the first fully rigorous one. The ray-casting test for
whether a point lies inside a polygon follows directly from the
theorem: count how many times a ray from the point crosses the
polygon's boundary, and an odd count means inside. Moshe Shimrat
published it as a computer algorithm in 1962.

*Implementation:* :func:`mathematicskit.geometry.systems.intersections.point_in_polygon`
implements this ray-casting test, and this domain's tests check it on a
concave, L-shaped polygon.
:func:`~mathematicskit.geometry.systems.intersections.segment_intersection`
implements the companion segment-intersection primitive by applying
Cramer's rule to the two segments' parametric equations.

*References:* C. Jordan, *Cours d'analyse de l'École polytechnique*,
vol. 3 (Paris: Gauthier-Villars, 1887); O. Veblen, "Theory on Plane
Curves in Non-Metrical Analysis Situs," Transactions of the American
Mathematical Society 6(1) (1905), 83-98; M. Shimrat, "Algorithm 112:
Position of Point Relative to Polygon," Communications of the ACM 5(8)
(1962), 434.

.. minigallery:: ../../examples/geometry/intersections/plot_01_segments_and_polygons.py

1908 -- Voronoi Diagrams
------------------------

Georgy Voronoi's 1908 paper partitioned space into regions, one per
input point, each containing every location closer to that point than
to any other. René Descartes had sketched the idea qualitatively in
1644, to describe how the influence of stars might divide space, and
Peter Gustav Lejeune Dirichlet had studied it rigorously in two and
three dimensions in 1850. The construction is now used far beyond
mathematics, from modeling crystal grain boundaries to drawing
service-area maps for a network of facilities.

*Implementation:* :func:`mathematicskit.geometry.systems.triangulation.voronoi_diagram`
computes this partition with :class:`scipy.spatial.Voronoi`.

*References:* G. Voronoi, "Nouvelles applications des paramètres
continus à la théorie des formes quadratiques. Deuxième mémoire:
Recherches sur les parallélloèdres primitifs," Journal für die reine
und angewandte Mathematik 134 (1908), 198-287.

.. minigallery:: ../../examples/geometry/triangulation/plot_01_delaunay_and_voronoi.py

1934 -- Delaunay Triangulation
------------------------------

Boris Delaunay's 1934 paper, building on Georgy Voronoi's 1908 diagrams
(above), defined the triangulation of a point set in which no point
lies inside any triangle's circumcircle. In the plane this is
equivalent to maximizing the smallest angle over all possible
triangulations. Delaunay triangulations therefore avoid the thin,
needle-like triangles that cause numerical trouble in interpolation and
finite-element meshing. They are also exactly the dual graph of the
corresponding Voronoi diagram.

*Implementation:* :func:`mathematicskit.geometry.systems.triangulation.delaunay_triangulation`
wraps :class:`scipy.spatial.Delaunay` (Qhull), and
:func:`~mathematicskit.geometry.systems.triangulation.voronoi_diagram`
wraps :class:`scipy.spatial.Voronoi` for its dual.

*References:* B. Delaunay, "Sur la sphère vide," Bulletin de l'Académie
des Sciences de l'URSS, Classe des Sciences Mathématiques et Naturelles
6 (1934), 793-800.

.. minigallery:: ../../examples/geometry/triangulation/plot_01_delaunay_and_voronoi.py

1972 -- Graham's Scan and the Convex Hull
-----------------------------------------

Ronald Graham's 1972 paper gave one of the first efficient convex-hull
algorithms. Sort the points by polar angle around a fixed pivot, then
sweep through them in order, discarding any point that would make the
hull-in-progress turn clockwise instead of counterclockwise. The
algorithm runs in :math:`O(n\log n)` time, dominated by the initial
sort. It remains one of the clearest examples of a global geometric
structure (the hull) built from a simple, local decision made at each
point.

*Implementation:* :func:`mathematicskit.geometry.systems.convex_hull.graham_scan`
implements this angular sweep as a teaching comparison. The primary
API is :func:`~mathematicskit.geometry.systems.convex_hull.convex_hull`,
which wraps :class:`scipy.spatial.ConvexHull` (the Qhull library).

*References:* R. L. Graham, "An Efficient Algorithm for Determining the
Convex Hull of a Finite Planar Set," Information Processing Letters
1(4) (1972), 132-133.

.. minigallery:: ../../examples/geometry/convex_hull/plot_01_qhull_vs_graham_scan.py

See Also
--------

- :doc:`/api/geometry`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/fractals_chaos_breakthroughs`
