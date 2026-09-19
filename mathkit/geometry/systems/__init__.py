"""Concrete computational-geometry algorithms."""

from mathkit.geometry.systems.convex_hull import convex_hull, graham_scan
from mathkit.geometry.systems.curves import frenet_serret_frame
from mathkit.geometry.systems.intersections import point_in_polygon, segment_intersection
from mathkit.geometry.systems.polygon import polygon_area, polygon_centroid
from mathkit.geometry.systems.triangulation import delaunay_triangulation, voronoi_diagram

__all__ = [
    "convex_hull",
    "graham_scan",
    "delaunay_triangulation",
    "voronoi_diagram",
    "segment_intersection",
    "point_in_polygon",
    "polygon_area",
    "polygon_centroid",
    "frenet_serret_frame",
]
