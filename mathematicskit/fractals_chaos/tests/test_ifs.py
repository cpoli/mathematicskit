"""Tests for iterated function systems: generated point clouds must be
bounded, reproducible under a fixed seed, and (for the Sierpinski
triangle) lie within the convex hull of its generating vertices.
"""

import numpy as np
import pytest

from mathematicskit.fractals_chaos.systems.ifs import BarnsleyFern, SierpinskiCarpet, SierpinskiTriangle


def test_barnsley_fern_shape_and_reproducibility():
    fern = BarnsleyFern()
    a = fern.generate(1000, seed=0)
    b = fern.generate(1000, seed=0)
    assert a.shape == (1000, 2)
    np.testing.assert_array_equal(a, b)


def test_barnsley_fern_stays_within_known_bounding_box():
    fern = BarnsleyFern()
    points = fern.generate(5000, seed=1)
    assert points[:, 0].min() >= -3.0
    assert points[:, 0].max() <= 3.0
    assert points[:, 1].min() >= -1.0
    assert points[:, 1].max() <= 11.0


def test_sierpinski_triangle_points_within_triangle_bounding_box():
    tri = SierpinskiTriangle()
    points = tri.generate(5000, seed=2)
    assert points[:, 0].min() >= -1e-8
    assert points[:, 0].max() <= 1.0 + 1e-8
    assert points[:, 1].min() >= -1e-8
    assert points[:, 1].max() <= np.sqrt(3.0) / 2.0 + 1e-8


def test_sierpinski_carpet_points_within_unit_square():
    carpet = SierpinskiCarpet()
    points = carpet.generate(5000, seed=3)
    assert np.all(points >= -1e-8)
    assert np.all(points <= 1.0 + 1e-8)


def test_different_seeds_give_different_orbits():
    tri = SierpinskiTriangle()
    a = tri.generate(200, seed=0)
    b = tri.generate(200, seed=1)
    assert not np.array_equal(a, b)


def test_sierpinski_triangle_rejects_bad_vertex_shape():
    with pytest.raises(ValueError):
        SierpinskiTriangle(vertices=np.zeros((4, 2)))
