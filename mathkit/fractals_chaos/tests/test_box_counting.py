"""Tests for box-counting dimension estimation against known/closed-form
dimensions: a line has dimension 1, a filled square has dimension ~2,
and the Sierpinski triangle/carpet IFS attractors match their known
fractal dimensions."""

import numpy as np
import pytest

from mathkit.fractals_chaos.systems.box_counting import box_counting_dimension
from mathkit.fractals_chaos.systems.ifs import SierpinskiCarpet, SierpinskiTriangle


def test_line_segment_has_dimension_one():
    x = np.linspace(0.0, 1.0, 20000)
    points = np.stack([x, np.zeros_like(x)], axis=1)
    result = box_counting_dimension(points)
    assert result.dimension == pytest.approx(1.0, abs=0.2)


def test_filled_square_has_dimension_near_two():
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, size=(20000, 2))
    result = box_counting_dimension(points)
    assert result.dimension == pytest.approx(2.0, abs=0.3)


def test_sierpinski_triangle_matches_known_dimension():
    points = SierpinskiTriangle().generate(60000, seed=1)
    result = box_counting_dimension(points)
    expected = np.log(3.0) / np.log(2.0)
    assert result.dimension == pytest.approx(expected, abs=0.2)


def test_sierpinski_carpet_matches_known_dimension():
    points = SierpinskiCarpet().generate(60000, seed=1)
    result = box_counting_dimension(points)
    expected = np.log(8.0) / np.log(3.0)
    assert result.dimension == pytest.approx(expected, abs=0.2)


def test_box_counts_are_non_increasing_as_box_size_grows():
    rng = np.random.default_rng(2)
    points = rng.uniform(0.0, 1.0, size=(10000, 2))
    result = box_counting_dimension(points)
    # box_sizes is generated largest-to-smallest, so counts should be non-decreasing.
    assert np.all(np.diff(result.box_counts) >= 0)


def test_rejects_degenerate_point_set():
    with pytest.raises(ValueError):
        box_counting_dimension(np.zeros((10, 2)))


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        box_counting_dimension(np.zeros((10, 3)))
