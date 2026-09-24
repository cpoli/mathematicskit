"""Tests for the enclosing circle, Hausdorff distance, Bézier curves,
Douglas-Peucker simplification, and the closest pair of points."""

from math import comb

import numpy as np
import pytest
from scipy.spatial import cKDTree

from mathematicskit.geometry.systems.bezier import bezier_curve, de_casteljau
from mathematicskit.geometry.systems.distances import hausdorff_distance
from mathematicskit.geometry.systems.enclosing import min_enclosing_circle
from mathematicskit.geometry.systems.proximity import closest_pair
from mathematicskit.geometry.systems.simplification import douglas_peucker


@pytest.mark.parametrize("seed", range(10))
def test_enclosing_circle_contains_all_points_and_is_tight(seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(300, 2))
    result = min_enclosing_circle(pts)
    distances = np.linalg.norm(pts - result.center, axis=1)
    assert np.all(distances <= result.radius + 1e-9)
    assert np.sum(np.isclose(distances, result.radius, atol=1e-9)) >= 2


def test_enclosing_circle_of_triangle_vertices():
    result = min_enclosing_circle([[0, 0], [2, 0], [1, np.sqrt(3)]])  # equilateral: circumcircle
    assert result.radius == pytest.approx(2 / np.sqrt(3))
    obtuse = min_enclosing_circle([[0, 0], [4, 0], [2, 0.5]])  # obtuse: diameter circle
    assert obtuse.radius == pytest.approx(2.0)


def test_enclosing_circle_order_independent():
    pts = np.random.default_rng(1).uniform(size=(100, 2))
    assert min_enclosing_circle(pts, seed=0).radius == pytest.approx(min_enclosing_circle(pts, seed=7).radius)


def test_hausdorff_distance_properties():
    a = np.random.default_rng(0).normal(size=(50, 2))
    assert hausdorff_distance(a, a) == 0.0
    assert hausdorff_distance(a, a + [3, 4]) == pytest.approx(5.0)
    b = np.vstack([a, [[10.0, 0.0]]])
    assert hausdorff_distance(a, b) == hausdorff_distance(b, a) > 5


def test_bezier_matches_bernstein_form():
    control = np.array([[0, 0], [1, 3], [3, -1], [4, 2]], dtype=float)
    t = np.linspace(0, 1, 11)
    n = len(control) - 1
    bernstein = sum(comb(n, i) * t[:, None] ** i * (1 - t[:, None]) ** (n - i) * control[i] for i in range(n + 1))
    assert np.allclose(bezier_curve(control, t), bernstein)


def test_bezier_interpolates_endpoints_and_level_shapes():
    control = [[0, 0], [1, 2], [3, 3], [4, 0]]
    curve = bezier_curve(control, [0.0, 1.0])
    assert np.allclose(curve, [[0, 0], [4, 0]])
    assert [len(level) for level in de_casteljau(control, 0.3)] == [4, 3, 2, 1]


def test_douglas_peucker_keeps_endpoints_and_respects_tolerance():
    x = np.linspace(0, 10, 500)
    line = np.column_stack([x, np.sin(x)])
    for eps in (0.01, 0.1, 0.5):
        simple = douglas_peucker(line, eps)
        assert np.allclose(simple[[0, -1]], line[[0, -1]])
        # every original point is within eps of the simplified polyline
        seg_d = []
        for p in line:
            d = min(np.linalg.norm(p - (a + np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1) * (b - a))) for a, b in zip(simple[:-1], simple[1:]))
            seg_d.append(d)
        assert max(seg_d) <= eps + 1e-12


def test_douglas_peucker_collapses_a_straight_line():
    line = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 2, 50)])
    assert len(douglas_peucker(line, 1e-9)) == 2


@pytest.mark.parametrize("seed", range(10))
def test_closest_pair_matches_kd_tree(seed):
    pts = np.random.default_rng(seed).uniform(size=(500, 2))
    result = closest_pair(pts)
    distances, _ = cKDTree(pts).query(pts, k=2)
    assert result.distance == pytest.approx(distances[:, 1].min())
    i, j = result.indices
    assert np.linalg.norm(pts[i] - pts[j]) == pytest.approx(result.distance)


def test_closest_pair_needs_two_points():
    with pytest.raises(ValueError):
        closest_pair([[0, 0]])
