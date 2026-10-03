"""Tests for the travelling-salesman heuristics and the Held-Karp algorithm."""

from itertools import permutations

import numpy as np
import pytest

from mathematicskit.optimization.systems.tsp import distance_matrix, tour_length, tsp_held_karp, tsp_nearest_neighbor, tsp_two_opt


@pytest.fixture
def cities():
    return np.random.default_rng(7).uniform(size=(9, 2))


def _brute_force(dist):
    n = dist.shape[0]
    return min(tour_length(dist, (0,) + p) for p in permutations(range(1, n)))


def test_held_karp_matches_brute_force(cities):
    d = distance_matrix(cities)
    result = tsp_held_karp(d)
    assert result.length == pytest.approx(_brute_force(d))
    assert sorted(result.tour) == list(range(9)) and result.tour[0] == 0
    assert result.length == pytest.approx(tour_length(d, result.tour))


def test_held_karp_asymmetric_distances():
    d = np.random.default_rng(1).uniform(1, 10, size=(7, 7))
    np.fill_diagonal(d, 0.0)
    assert tsp_held_karp(d).length == pytest.approx(_brute_force(d))


def test_regular_polygon_optimum_is_its_perimeter():
    n = 12
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])[np.random.default_rng(0).permutation(n)]
    d = distance_matrix(pts)
    perimeter = n * 2 * np.sin(np.pi / n)
    assert tsp_held_karp(d).length == pytest.approx(perimeter)
    assert tsp_two_opt(d).length == pytest.approx(perimeter)


def test_heuristics_bounded_below_by_optimum(cities):
    d = distance_matrix(cities)
    optimum = tsp_held_karp(d).length
    nn = tsp_nearest_neighbor(d)
    opt2 = tsp_two_opt(d, nn.tour)
    assert optimum <= opt2.length + 1e-12 <= nn.length + 1e-12
    assert opt2.history[0] == pytest.approx(nn.length)
    assert np.all(np.diff(opt2.history) < 0)


def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0


def test_two_opt_tour_has_no_crossings():
    pts = np.random.default_rng(3).uniform(size=(40, 2))
    tour = tsp_two_opt(distance_matrix(pts), list(range(40))).tour
    edges = [(pts[tour[i]], pts[tour[(i + 1) % 40]]) for i in range(40)]
    for i in range(40):
        for j in range(i + 2, 40):
            if i == 0 and j == 39:
                continue
            assert not _segments_cross(*edges[i], *edges[j])


def test_tiny_instances():
    assert tsp_held_karp(np.zeros((1, 1))).tour == [0]
    two = distance_matrix([[0, 0], [0, 2]])
    assert tsp_held_karp(two).length == pytest.approx(4.0)
    assert tsp_nearest_neighbor(two, start=1).tour == [1, 0]
