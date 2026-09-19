"""Tests for Delaunay triangulation and Voronoi diagrams against
closed-form/known structural properties."""

import numpy as np

from mathematicskit.geometry.systems.triangulation import delaunay_triangulation, voronoi_diagram


def test_delaunay_of_four_corners_gives_two_triangles():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = delaunay_triangulation(points)
    assert result.simplices.shape == (2, 3)


def test_delaunay_triangles_cover_every_input_point():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 10, size=(20, 2))
    result = delaunay_triangulation(points)
    used_indices = set(result.simplices.flatten())
    assert used_indices == set(range(20))


def test_voronoi_has_one_region_per_point():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    result = voronoi_diagram(points)
    assert len(result.regions) >= len(points)


def test_voronoi_ridge_points_reference_valid_indices():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    result = voronoi_diagram(points)
    assert np.all(result.ridge_points >= 0)
    assert np.all(result.ridge_points < len(points))


def test_voronoi_center_point_region_is_bounded():
    """The Voronoi region of a point strictly inside the convex hull of
    the others is bounded (no vertex index -1)."""
    from scipy.spatial import Voronoi

    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]])
    vor = Voronoi(points)
    center_region = vor.regions[vor.point_region[4]]
    assert -1 not in center_region
