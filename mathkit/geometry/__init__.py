"""mathkit.geometry: computational geometry, built directly on scipy.spatial.

Convex hull via ``scipy.spatial.ConvexHull``, with a hand-rolled Graham
scan kept as a pedagogical comparison in 2D; Delaunay triangulation via
``scipy.spatial.Delaunay`` and its dual Voronoi diagram via
``scipy.spatial.Voronoi``; line/segment intersection and point-in-
polygon tests (hand-rolled -- no direct scipy equivalent); polygon area
and centroid via the shoelace formula; and curvature, arc length, and
the Frenet-Serret frame for parametric plane/space curves (derivatives
via ``numpy.gradient``, arc length via ``scipy.integrate``, the frame
itself assembled by mathkit's own code).
"""

from mathkit.geometry.core.base import ConvexHullResult, CurveFrameResult, TriangulationResult, VoronoiResult
from mathkit.geometry.systems.convex_hull import convex_hull, graham_scan
from mathkit.geometry.systems.curves import frenet_serret_frame
from mathkit.geometry.systems.intersections import point_in_polygon, segment_intersection
from mathkit.geometry.systems.polygon import polygon_area, polygon_centroid
from mathkit.geometry.systems.triangulation import delaunay_triangulation, voronoi_diagram
from mathkit.geometry.utils.curves_library import circle, ellipse, helix

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
]
