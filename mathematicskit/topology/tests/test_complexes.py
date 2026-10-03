"""Tests for mathematicskit.topology.core.base and systems.complexes."""

import itertools
from math import comb

import numpy as np
import pytest

from mathematicskit.topology import (
    SimplicialComplex,
    barycentric_subdivision,
    betti_numbers,
    circle,
    klein_bottle,
    mobius_strip,
    projective_plane,
    simplex,
    simplicial_product,
    sphere,
    torus,
)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_simplex_has_binomial_face_counts_and_euler_characteristic_one(n):
    K = simplex(n)
    assert K.f_vector == tuple(comb(n + 1, k + 1) for k in range(n + 1))
    assert K.euler_characteristic == 1


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["simplex", "cross_polytope"])
def test_sphere_euler_characteristic_is_one_plus_minus_one(n, kind):
    assert sphere(n, kind).euler_characteristic == 1 + (-1) ** n


def test_sphere_rejects_unknown_kind_and_circle_needs_three_vertices():
    with pytest.raises(ValueError):
        sphere(2, kind="cube")
    with pytest.raises(ValueError):
        circle(2)
    with pytest.raises(ValueError):
        torus(2, 3)


@pytest.mark.parametrize("K", [sphere(3), torus(3, 3), klein_bottle(4, 3), projective_plane(), mobius_strip(5)])
def test_boundary_of_boundary_is_zero(K):
    for k in range(2, K.dimension + 1):
        assert np.all(K.boundary_matrix(k - 1) @ K.boundary_matrix(k) == 0)


def test_boundary_matrix_shapes_including_the_zero_maps():
    K = torus(3, 3)
    assert K.boundary_matrix(0).shape == (0, 9)
    assert K.boundary_matrix(3).shape == (18, 0)


def test_closure_union_intersection_and_containment():
    A = SimplicialComplex([[0, 1, 2]])
    B = SimplicialComplex([[1, 2, 3]])
    assert (1, 0) in A and [0, 1, 2] in A and (0, 3) not in A and () not in A
    assert (A & B) == SimplicialComplex([[1, 2]])
    assert (A | B).f_vector == (4, 5, 2)
    assert len(A) == 7 and list(A)[0] == (0,)
    assert A.maximal_simplices == [(0, 1, 2)]
    assert SimplicialComplex([[0, 1], [2]]).maximal_simplices == [(2,), (0, 1)]
    assert A.skeleton(1).f_vector == (3, 3) and A.skeleton(-1).dimension == -1
    assert SimplicialComplex().euler_characteristic == 0 and SimplicialComplex().vertices == []
    assert A.simplices(5) == [] and "dimension=2" in repr(A)


def test_link_of_a_torus_vertex_is_a_hexagon():
    link = torus(4, 4).link(0)
    assert link.f_vector == (6, 6)
    assert betti_numbers(link) == (1, 1)


@pytest.mark.parametrize("K", [simplex(2), torus(3, 3), projective_plane(), sphere(2)])
def test_barycentric_subdivision_preserves_euler_characteristic_and_homology(K):
    sd = barycentric_subdivision(K)
    assert sd.euler_characteristic == K.euler_characteristic
    assert betti_numbers(sd) == betti_numbers(K)
    assert sd.f_vector[0] == len(K)


def test_barycentric_subdivision_places_vertices_at_barycenters():
    K = simplex(2)
    sd = barycentric_subdivision(K)
    assert np.allclose(sd.coordinates[-1], K.coordinates.mean(axis=0))


@pytest.mark.parametrize(("p", "q"), [(1, 1), (1, 2), (2, 2)])
def test_product_of_simplices_has_binomial_top_simplices(p, q):
    assert simplicial_product(simplex(p), simplex(q)).f_vector[-1] == comb(p + q, p)


def test_kunneth_formula_for_products_of_circles_and_spheres():
    def convolve(a, b):
        return tuple(sum(a[i] * b[k - i] for i in range(len(a)) if 0 <= k - i < len(b)) for k in range(len(a) + len(b) - 1))

    for K, L in itertools.product([circle(3), circle(4)], [circle(3), sphere(2)]):
        assert betti_numbers(simplicial_product(K, L)) == convolve(betti_numbers(K), betti_numbers(L))
    assert simplicial_product(circle(3), circle(4)).coordinates.shape == (12, 4)
