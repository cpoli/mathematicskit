"""Tests for polygon area/centroid (shoelace formula) against
closed-form/known results."""

import numpy as np
import pytest

from mathematicskit.geometry.systems.polygon import polygon_area, polygon_centroid


def test_rectangle_area():
    rect = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    assert polygon_area(rect) == pytest.approx(12.0)


def test_triangle_area_matches_half_base_times_height():
    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert polygon_area(triangle) == pytest.approx(0.5 * 4.0 * 3.0)


def test_area_is_orientation_independent():
    ccw = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    cw = ccw[::-1]
    assert polygon_area(ccw) == pytest.approx(polygon_area(cw))


def test_rectangle_centroid_is_geometric_center():
    rect = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    np.testing.assert_allclose(polygon_centroid(rect), [2.0, 1.5])


def test_triangle_centroid_matches_average_of_vertices():
    """A triangle's centroid via the shoelace formula equals the simple
    average of its vertices -- true only for triangles, not general
    polygons."""
    triangle = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 9.0]])
    expected = triangle.mean(axis=0)
    np.testing.assert_allclose(polygon_centroid(triangle), expected)


def test_regular_hexagon_area_matches_closed_form():
    n = 6
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    side = 1.0
    hexagon = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    expected = (3.0 * np.sqrt(3.0) / 2.0) * side**2
    assert polygon_area(hexagon) == pytest.approx(expected, rel=1e-6)
