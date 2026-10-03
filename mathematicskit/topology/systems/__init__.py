"""Concrete topology algorithms."""

from mathematicskit.topology.systems.complexes import (
    barycentric_subdivision,
    circle,
    klein_bottle,
    mobius_strip,
    projective_plane,
    simplex,
    simplicial_product,
    sphere,
    torus,
)
from mathematicskit.topology.systems.curves import linking_number, turning_number
from mathematicskit.topology.systems.fixed_points import (
    brouwer_fixed_point,
    fully_labeled_triangles,
    is_sperner_labeling,
    lefschetz_number,
    random_sperner_labeling,
    triangle_grid,
    vector_field_index,
)
from mathematicskit.topology.systems.fundamental_group import abelianization, fundamental_group
from mathematicskit.topology.systems.homology import betti_numbers, homology, mayer_vietoris, smith_normal_form
from mathematicskit.topology.systems.persistence import bottleneck_distance, persistent_homology
from mathematicskit.topology.systems.point_clouds import (
    cech_complex,
    lower_star_filtration,
    mapper_graph,
    vietoris_rips_complex,
    vietoris_rips_filtration,
)
from mathematicskit.topology.systems.surfaces import boundary_components, classify_surface, critical_points, is_orientable

__all__ = [
    "simplex",
    "sphere",
    "circle",
    "torus",
    "klein_bottle",
    "mobius_strip",
    "projective_plane",
    "barycentric_subdivision",
    "simplicial_product",
    "smith_normal_form",
    "betti_numbers",
    "homology",
    "mayer_vietoris",
    "fundamental_group",
    "abelianization",
    "is_orientable",
    "boundary_components",
    "classify_surface",
    "critical_points",
    "triangle_grid",
    "is_sperner_labeling",
    "random_sperner_labeling",
    "fully_labeled_triangles",
    "brouwer_fixed_point",
    "vector_field_index",
    "lefschetz_number",
    "linking_number",
    "turning_number",
    "vietoris_rips_complex",
    "vietoris_rips_filtration",
    "cech_complex",
    "lower_star_filtration",
    "mapper_graph",
    "persistent_homology",
    "bottleneck_distance",
]
