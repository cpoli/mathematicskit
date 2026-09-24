"""mathematicskit.geometry: computational geometry, built directly on scipy.spatial.

Convex hull via ``scipy.spatial.ConvexHull``, with a hand-rolled Graham
scan kept as a pedagogical comparison in 2D; Delaunay triangulation via
``scipy.spatial.Delaunay`` and its dual Voronoi diagram via
``scipy.spatial.Voronoi``; line/segment intersection and point-in-
polygon tests (hand-rolled -- no direct scipy equivalent); polygon area
and centroid via the shoelace formula; and curvature, arc length, and
the Frenet-Serret frame for parametric plane/space curves (derivatives
via ``numpy.gradient``, arc length via ``scipy.integrate``, the frame
itself assembled by mathematicskit's own code); Heron's formula and
Pick's lattice-point counts; Euler's polyhedron formula on 3D hulls;
Gaussian and mean curvature of parametric surfaces; the smallest
enclosing circle (Welzl); the Hausdorff distance; Bézier curves via de
Casteljau; Douglas-Peucker line simplification; and the closest pair of
points by divide and conquer.
"""

from mathematicskit.geometry.core.base import (
    CircleResult,
    ClosestPairResult,
    ConvexHullResult,
    CurveFrameResult,
    LatticePolygonResult,
    PolyhedronResult,
    SurfaceCurvatureResult,
    TriangulationResult,
    VoronoiResult,
)
from mathematicskit.geometry.systems.bezier import bezier_curve, de_casteljau
from mathematicskit.geometry.systems.convex_hull import convex_hull, graham_scan
from mathematicskit.geometry.systems.curves import frenet_serret_frame
from mathematicskit.geometry.systems.distances import hausdorff_distance
from mathematicskit.geometry.systems.enclosing import min_enclosing_circle
from mathematicskit.geometry.systems.intersections import point_in_polygon, segment_intersection
from mathematicskit.geometry.systems.polygon import heron_area, lattice_point_counts, polygon_area, polygon_centroid
from mathematicskit.geometry.systems.polyhedra import polyhedron_counts
from mathematicskit.geometry.systems.proximity import closest_pair
from mathematicskit.geometry.systems.simplification import douglas_peucker
from mathematicskit.geometry.systems.surfaces import cylinder_surface, sphere_surface, surface_curvature, torus_surface
from mathematicskit.geometry.systems.triangulation import delaunay_triangulation, voronoi_diagram
from mathematicskit.geometry.utils.curves_library import circle, ellipse, helix

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ConvexHullResult",
    "TriangulationResult",
    "VoronoiResult",
    "CurveFrameResult",
    "convex_hull",
    "graham_scan",
    "delaunay_triangulation",
    "voronoi_diagram",
    "segment_intersection",
    "point_in_polygon",
    "polygon_area",
    "polygon_centroid",
    "frenet_serret_frame",
    "circle",
    "helix",
    "ellipse",
    "SurfaceCurvatureResult",
    "PolyhedronResult",
    "CircleResult",
    "LatticePolygonResult",
    "ClosestPairResult",
    "bezier_curve",
    "de_casteljau",
    "hausdorff_distance",
    "min_enclosing_circle",
    "polyhedron_counts",
    "closest_pair",
    "douglas_peucker",
    "cylinder_surface",
    "sphere_surface",
    "surface_curvature",
    "torus_surface",
    "heron_area",
    "lattice_point_counts",
]
