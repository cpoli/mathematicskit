"""Tests for convex hull (scipy and hand-rolled Graham scan) against
closed-form/known results."""

import numpy as np
import pytest

from mathkit.geometry.systems.convex_hull import convex_hull, graham_scan


def _square_with_interior_point():
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])


def test_convex_hull_excludes_interior_point():
    result = convex_hull(_square_with_interior_point())
    assert sorted(result.vertices) == [0, 1, 2, 3]


def test_convex_hull_area_matches_unit_square():
    result = convex_hull(_square_with_interior_point())
    assert result.volume == pytest.approx(1.0)


def test_graham_scan_matches_scipy_convex_hull():
    points = _square_with_interior_point()
    scipy_result = convex_hull(points)
    graham_result = graham_scan(points)
    assert sorted(int(v) for v in graham_result.vertices) == sorted(int(v) for v in scipy_result.vertices)
    assert graham_result.volume == pytest.approx(scipy_result.volume)


def test_graham_scan_on_random_points_matches_scipy():
    rng = np.random.default_rng(0)
    points = rng.uniform(-5, 5, size=(30, 2))
    scipy_result = convex_hull(points)
    graham_result = graham_scan(points)
    assert set(int(v) for v in graham_result.vertices) == set(int(v) for v in scipy_result.vertices)


def test_graham_scan_rejects_too_few_points():
    with pytest.raises(ValueError):
        graham_scan(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_convex_hull_of_triangle_is_all_three_points():
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = convex_hull(triangle)
    assert len(result.vertices) == 3
