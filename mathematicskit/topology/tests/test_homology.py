"""Tests for mathematicskit.topology.systems.homology and utils."""

import numpy as np
import pytest

from mathematicskit.topology import (
    betti_numbers,
    circle,
    homology,
    klein_bottle,
    mayer_vietoris,
    mobius_strip,
    planar_betti_numbers,
    projective_plane,
    rank_mod_p,
    simplex,
    smith_normal_form,
    sphere,
    torus,
)


def _is_unimodular(M):
    return round(abs(np.linalg.det(M.astype(float)))) == 1


@pytest.mark.parametrize("seed", range(6))
def test_smith_normal_form_is_diagonal_divisible_and_unimodular(seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(-6, 7, size=(rng.integers(1, 6), rng.integers(1, 6)))
    result = smith_normal_form(A)
    assert np.array_equal(result.U @ A @ result.V, result.D)
    assert _is_unimodular(result.U) and _is_unimodular(result.V)
    assert np.count_nonzero(result.D - np.diag(np.diagonal(result.D))) == 0 if result.D.shape[0] == result.D.shape[1] else True
    d = result.invariant_factors
    assert all(x > 0 for x in d) and all(b % a == 0 for a, b in zip(d, d[1:], strict=False))
    assert result.rank == np.linalg.matrix_rank(A)


def test_smith_normal_form_determinant_and_empty_matrix():
    A = np.array([[4, 6], [2, 8]])
    assert np.prod(smith_normal_form(A).invariant_factors) == abs(round(np.linalg.det(A)))
    assert smith_normal_form(np.zeros((0, 3), dtype=int)).D.shape == (0, 3)


@pytest.mark.parametrize(
    ("K", "expected"),
    [
        (simplex(3), "H_0 = Z, H_1 = 0, H_2 = 0, H_3 = 0"),
        (sphere(2), "H_0 = Z, H_1 = 0, H_2 = Z"),
        (torus(3, 3), "H_0 = Z, H_1 = Z^2, H_2 = Z"),
        (klein_bottle(3, 3), "H_0 = Z, H_1 = Z + Z/2, H_2 = 0"),
        (projective_plane(), "H_0 = Z, H_1 = Z/2, H_2 = 0"),
        (mobius_strip(5), "H_0 = Z, H_1 = Z, H_2 = 0"),
    ],
)
def test_integer_homology_of_standard_spaces(K, expected):
    result = homology(K)
    assert str(result) == expected
    assert result.euler_characteristic == K.euler_characteristic
    assert result.group(9) == "0"


@pytest.mark.parametrize("n", [1, 2, 3])
def test_spheres_have_betti_numbers_one_zero_one(n):
    assert betti_numbers(sphere(n)) == (1,) + (0,) * (n - 1) + (1,)


def test_mod_two_betti_numbers_see_torsion_but_rational_ones_do_not():
    assert betti_numbers(projective_plane()) == (1, 0, 0)
    assert betti_numbers(projective_plane(), field=2) == (1, 1, 1)
    assert betti_numbers(projective_plane(), field=3) == (1, 0, 0)
    assert betti_numbers(klein_bottle(), field=2) == (1, 2, 1)


def test_rank_mod_p_matches_rational_rank_for_unimodular_and_drops_for_divisible():
    assert rank_mod_p(np.eye(4, dtype=int), 5) == 4
    assert rank_mod_p([[3, 6], [1, 2]], 3) == 1
    assert rank_mod_p(np.zeros((0, 2), dtype=int), 2) == 0


def test_mayer_vietoris_predicts_the_torus_from_two_annuli():
    T = torus(8, 4)
    A = T.induced_subcomplex([v for v in T.vertices if v // 4 in range(5)])
    B = T.induced_subcomplex([v for v in T.vertices if v // 4 in (4, 5, 6, 7, 0)])
    result = mayer_vietoris(A, B)
    assert result.betti_intersection == (2, 2)
    assert result.betti_union == (1, 2, 1)
    assert result.predicted_union == result.betti_union


def test_mayer_vietoris_with_disjoint_pieces():
    C = circle(6)
    A, B = C.induced_subcomplex([0, 1]), C.induced_subcomplex([3, 4])
    result = mayer_vietoris(A, B)
    assert result.betti_union == (2, 0) and result.predicted_union == (2, 0)


def test_planar_betti_numbers_of_an_annulus_and_two_blobs():
    x, y = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
    r = np.hypot(x, y)
    assert planar_betti_numbers((r > 0.4) & (r < 0.8)) == (1, 1)
    assert planar_betti_numbers((np.hypot(x - 0.5, y) < 0.3) | (np.hypot(x + 0.5, y) < 0.3)) == (2, 0)


def test_smith_normal_form_restores_divisibility_of_coprime_diagonal():
    result = smith_normal_form([[2, 0], [0, 3]])
    assert result.invariant_factors == [1, 6]
    assert np.array_equal(result.U @ np.array([[2, 0], [0, 3]]) @ result.V, result.D)


def test_rank_mod_p_stops_once_every_row_has_a_pivot():
    assert rank_mod_p([[1, 0, 0]], 7) == 1
