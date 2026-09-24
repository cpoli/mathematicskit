"""Tests for Heron's formula, Pick's theorem, and Euler's polyhedron formula."""

import itertools

import numpy as np
import pytest

from mathematicskit.geometry.systems.polygon import heron_area, lattice_point_counts, polygon_area
from mathematicskit.geometry.systems.polyhedra import polyhedron_counts


def test_heron_matches_shoelace_on_random_triangles():
    rng = np.random.default_rng(0)
    for _ in range(50):
        tri = rng.normal(size=(3, 2))
        a, b, c = (np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3))
        assert heron_area(a, b, c) == pytest.approx(polygon_area(tri), rel=1e-10)


def test_heron_is_stable_for_needle_triangles():
    # Kahan's example: a = b = 1e5, c = 1e-5 area is c / 2 * sqrt(a^2 - c^2/4) ~ 0.5
    assert heron_area(1e5, 1e5, 1e-5) == pytest.approx(0.5, rel=1e-9)


def test_heron_rejects_impossible_triangle():
    with pytest.raises(ValueError):
        heron_area(1, 1, 3)


@pytest.mark.parametrize(
    "vertices",
    [
        [[0, 0], [4, 0], [4, 3], [0, 3]],
        [[0, 0], [5, 0], [2, 7]],
        [[0, 0], [6, 0], [6, 2], [3, 5], [0, 2]],
        [[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]],  # non-convex
    ],
)
def test_pick_theorem(vertices):
    result = lattice_point_counts(vertices)
    assert result.pick_area == pytest.approx(result.area)


def _platonic_solids():
    phi = (1 + 5**0.5) / 2
    cube = list(itertools.product((-1, 1), repeat=3))
    tetra = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    octa = [p for i in range(3) for s in (-1, 1) for p in [tuple(s if k == i else 0 for k in range(3))]]
    icosa = [p for a, b in itertools.product((-1, 1), repeat=2) for p in ((0, a, b * phi), (a, b * phi, 0), (b * phi, 0, a))]
    dodeca = cube + [p for a, b in itertools.product((-1, 1), repeat=2) for p in ((0, a / phi, b * phi), (a / phi, b * phi, 0), (b * phi, 0, a / phi))]
    return {
        "tetrahedron": (tetra, 4, 6, 4),
        "cube": (cube, 8, 12, 6),
        "octahedron": (octa, 6, 12, 8),
        "dodecahedron": (dodeca, 20, 30, 12),
        "icosahedron": (icosa, 12, 30, 20),
    }


@pytest.mark.parametrize("name", list(_platonic_solids()))
def test_platonic_solids_satisfy_euler_formula(name):
    points, v, e, f = _platonic_solids()[name]
    result = polyhedron_counts(np.array(points, dtype=float))
    assert (result.vertices, result.edges, result.faces) == (v, e, f)
    assert result.euler_characteristic == 2


def test_random_convex_hull_has_euler_characteristic_two():
    rng = np.random.default_rng(3)
    assert polyhedron_counts(rng.normal(size=(200, 3))).euler_characteristic == 2
