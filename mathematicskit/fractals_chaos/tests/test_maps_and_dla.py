"""Tests for the Hénon map and diffusion-limited aggregation."""

import numpy as np
import pytest

from mathematicskit.fractals_chaos.systems.box_counting import box_counting_dimension
from mathematicskit.fractals_chaos.systems.dla import dla_cluster
from mathematicskit.fractals_chaos.systems.maps import henon_map


def test_henon_orbit_satisfies_the_map():
    orbit = henon_map(500, discard=0)
    x, y = orbit[:-1].T
    assert np.allclose(orbit[1:, 0], 1 - 1.4 * x**2 + y)
    assert np.allclose(orbit[1:, 1], 0.3 * x)


def test_henon_attractor_is_bounded_and_fractal():
    orbit = henon_map(100000)
    assert np.all(np.abs(orbit) < 1.5)
    assert 1.1 < box_counting_dimension(orbit).dimension < 1.45


def test_henon_jacobian_determinant_is_minus_b():
    a, b = 1.4, 0.3
    x = 0.37
    jacobian = np.array([[-2 * a * x, 1.0], [b, 0.0]])
    assert np.linalg.det(jacobian) == pytest.approx(-b)


def test_dla_cluster_is_connected_and_unique():
    cluster = dla_cluster(1500, seed=3)
    cells = {tuple(p) for p in cluster}
    assert len(cells) == len(cluster)
    for x, y in cluster[1:]:
        assert any((x + dx, y + dy) in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def test_dla_mass_radius_dimension_near_1_7():
    cluster = dla_cluster(8000, seed=0)
    r = np.sqrt((cluster**2).sum(axis=1))
    radii = np.logspace(np.log10(5), np.log10(r.max() / 2), 10)
    counts = [(r < rho).sum() for rho in radii]
    slope = np.polyfit(np.log(radii), np.log(counts), 1)[0]
    assert slope == pytest.approx(1.71, abs=0.12)


def test_dla_is_reproducible():
    assert np.array_equal(dla_cluster(300, seed=5), dla_cluster(300, seed=5))
