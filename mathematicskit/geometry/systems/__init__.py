"""Concrete computational-geometry algorithms."""

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
