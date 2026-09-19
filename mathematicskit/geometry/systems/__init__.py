"""Concrete computational-geometry algorithms."""

from mathematicskit.geometry.systems.convex_hull import convex_hull, graham_scan
from mathematicskit.geometry.systems.curves import frenet_serret_frame
from mathematicskit.geometry.systems.intersections import point_in_polygon, segment_intersection
from mathematicskit.geometry.systems.polygon import polygon_area, polygon_centroid
from mathematicskit.geometry.systems.triangulation import delaunay_triangulation, voronoi_diagram

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
