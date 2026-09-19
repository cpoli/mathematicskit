"""Tests for segment intersection and point-in-polygon against
closed-form/known results."""

import numpy as np
import pytest

from mathkit.geometry.systems.intersections import point_in_polygon, segment_intersection


def test_crossing_segments_intersect_at_known_point():
    p = segment_intersection(np.array([0.0, 0.0]), np.array([2.0, 2.0]), np.array([0.0, 2.0]), np.array([2.0, 0.0]))
    np.testing.assert_allclose(p, [1.0, 1.0])


def test_parallel_segments_do_not_intersect():
    p = segment_intersection(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert p is None


def test_non_overlapping_segments_do_not_intersect():
    p = segment_intersection(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, -1.0]), np.array([2.0, 1.0]))
    assert p is None


def test_segments_touching_at_endpoint():
    p = segment_intersection(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(p, [1.0, 0.0])


@pytest.mark.parametrize(
    "point,expected",
    [
        ((2.0, 2.0), True),
        ((5.0, 2.0), False),
        ((-1.0, -1.0), False),
        ((0.1, 0.1), True),
    ],
)
def test_point_in_polygon_square(point, expected):
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    assert point_in_polygon(np.array(point), square) == expected


def test_point_in_concave_polygon():
    # An "L"-shaped (concave) polygon.
    l_shape = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    assert point_in_polygon(np.array([0.5, 0.5]), l_shape)  # inside the "foot"
    assert not point_in_polygon(np.array([1.5, 1.5]), l_shape)  # inside the notch, outside the L
