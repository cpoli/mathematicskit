"""Tests for mathematicskit.topology.systems.surfaces and fundamental_group."""

import numpy as np
import pytest

from mathematicskit.topology import (
    SimplicialComplex,
    abelianization,
    barycentric_subdivision,
    boundary_components,
    circle,
    classify_surface,
    critical_points,
    fundamental_group,
    homology,
    is_orientable,
    klein_bottle,
    mobius_strip,
    projective_plane,
    simplex,
    sphere,
    torus,
)
from mathematicskit.topology.systems.fundamental_group import _free_reduce


@pytest.mark.parametrize(
    ("K", "name", "orientable", "genus", "b"),
    [
        (sphere(2), "sphere", True, 0, 0),
        (simplex(2), "disk", True, 0, 1),
        (torus(), "torus", True, 1, 0),
        (klein_bottle(), "Klein bottle", False, 2, 0),
        (projective_plane(), "projective plane", False, 1, 0),
        (mobius_strip(), "Möbius strip", False, 1, 1),
        (mobius_strip(6, 2), "Möbius strip", False, 1, 1),
    ],
)
def test_classification_of_standard_surfaces(K, name, orientable, genus, b):
    result = classify_surface(K)
    assert (result.name, result.orientable, result.genus, result.boundary_components) == (name, orientable, genus, b)


def test_classification_of_annulus_and_unnamed_surfaces():
    T = torus(6, 4)
    annulus = T.induced_subcomplex([v for v in T.vertices if v // 4 in range(4)])
    assert classify_surface(annulus).name == "annulus"
    punctured_torus = SimplicialComplex([t for t in T.simplices(2) if t != T.simplices(2)[0]])
    assert classify_surface(punctured_torus).name == "orientable surface of genus 1 with 1 boundary component"
    punctured_klein = SimplicialComplex([t for t in klein_bottle().simplices(2)][1:])
    assert classify_surface(punctured_klein).name == "connected sum of 2 projective planes with 1 boundary component"


def test_classify_surface_rejects_non_surfaces():
    with pytest.raises(ValueError):
        classify_surface(circle(5))
    with pytest.raises(ValueError):
        classify_surface(SimplicialComplex([[0, 1, 2], [3, 4, 5]]))
    with pytest.raises(ValueError):
        classify_surface(SimplicialComplex([[0, 1, 2], [0, 1, 3], [0, 1, 4]]))
    with pytest.raises(ValueError):
        classify_surface(SimplicialComplex([[0, 1, 2], [0, 3, 4]]))
    with pytest.raises(ValueError):
        is_orientable(SimplicialComplex([[0, 1, 2], [2, 3]]))


def test_closed_orientable_iff_top_homology_is_z():
    for K in (sphere(2), torus(), klein_bottle(), projective_plane(), sphere(3)):
        assert is_orientable(K) == (homology(K).betti_numbers[-1] == 1)
    assert boundary_components(torus()) == 0


@pytest.mark.parametrize(
    "K", [sphere(2), torus(3, 3), klein_bottle(3, 3), projective_plane(), mobius_strip(5), circle(4), barycentric_subdivision(projective_plane())]
)
def test_abelianized_fundamental_group_is_first_homology(K):
    assert abelianization(fundamental_group(K)) == homology(K).group(1)


def test_fundamental_group_presentations():
    assert str(fundamental_group(projective_plane())) == "< a | aa >"
    assert str(fundamental_group(sphere(2))) == "1"
    assert str(fundamental_group(circle(5))) == "< a >"
    torus_group = fundamental_group(torus(3, 3))
    assert len(torus_group.generators) == 2 and sorted(e for _, e in torus_group.relations[0]) == [-1, -1, 1, 1]
    raw = fundamental_group(torus(3, 3), simplify=False)
    assert len(raw.generators) == 27 - 8 and abelianization(raw) == "Z^2"
    assert _free_reduce([(0, 1), (1, 1), (1, -1), (0, -1)]) == []
    assert _free_reduce([(0, 1), (1, 1), (0, -1)]) == [(1, 1)]


def test_fundamental_group_of_a_wedge_of_many_circles_names_generators_past_z():
    wedge = SimplicialComplex([[0, v] for v in range(1, 30)] + [[1, v] for v in range(2, 30)])
    P = fundamental_group(wedge)
    assert len(P.generators) == 28 and P.relations == []
    assert P.word([(27, 1), (27, -1)]) == "g27G27"
    assert abelianization(P) == homology(wedge).group(1) == "Z^28"


def test_banchoff_indices_sum_to_euler_characteristic_and_bound_betti_numbers():
    rng = np.random.default_rng(1)
    for K in (torus(8, 6), klein_bottle(6, 5), sphere(2, "cross_polytope"), projective_plane()):
        result = critical_points(K, rng.normal(size=len(K.vertices)))
        assert result.euler_characteristic == K.euler_characteristic
        assert len(result.minima) >= 1 and len(result.maxima) >= 1


def test_upright_torus_height_has_four_critical_points():
    T = torus(8, 6)
    result = critical_points(T, T.coordinates[:, 0] + 1e-3 * T.coordinates[:, 2])
    assert (len(result.minima), len(result.saddles), len(result.maxima)) == (1, 2, 1)
    assert np.count_nonzero(result.index) == 4
