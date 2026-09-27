"""Tests for Möbius transformations, the Joukowski map, and mapped grids."""

import numpy as np
import pytest

from mathematicskit.complex_analysis.systems.conformal_maps import classify_mobius, joukowski_map, map_grid, mobius_transform
from mathematicskit.complex_analysis.systems.holomorphic import complex_derivative


def test_cayley_transform_maps_upper_half_plane_to_unit_disk():
    cayley = lambda z: mobius_transform(z, 1, -1j, 1, 1j)
    assert cayley(1j) == pytest.approx(0)
    assert np.abs(cayley(np.linspace(-50, 50, 101))) == pytest.approx(np.ones(101))
    rng = np.random.default_rng(0)
    upper = rng.normal(size=200) + 1j * rng.uniform(0.01, 5, size=200)
    assert np.all(np.abs(cayley(upper)) < 1)


def test_mobius_composition_is_matrix_product_and_preserves_cross_ratio():
    z = np.array([0.3 + 0.1j, -1 + 2j, 2.5, 1j])
    m1, m2 = (2, 1j, 1, 3), (1, -1, 1j, 2)
    composed = np.array(m1).reshape(2, 2) @ np.array(m2).reshape(2, 2)
    assert mobius_transform(mobius_transform(z, *m2), *m1) == pytest.approx(mobius_transform(z, *composed.ravel()))

    def cross_ratio(a, b, c, d):
        return (a - c) * (b - d) / ((a - d) * (b - c))

    assert cross_ratio(*mobius_transform(z, *m1)) == pytest.approx(cross_ratio(*z))


def test_singular_mobius_raises():
    with pytest.raises(ValueError):
        mobius_transform(1.0, 1, 2, 2, 4)


def test_joukowski_flattens_circle_and_scales_with_c():
    theta = np.linspace(0, 2 * np.pi, 50)
    assert joukowski_map(np.exp(1j * theta)) == pytest.approx(2 * np.cos(theta))
    assert joukowski_map(2 * np.exp(1j * theta), c=2) == pytest.approx(4 * np.cos(theta))


def test_conformal_map_preserves_angle_between_directions():
    """The image of a small right angle under z^2 + z (f' != 0) is still a right angle."""
    f = lambda z: z**2 + z
    z0, h = 0.4 + 0.3j, 1e-6
    d1, d2 = f(z0 + h) - f(z0), f(z0 + 1j * h) - f(z0)
    assert abs(np.angle(d2 / d1)) == pytest.approx(np.pi / 2, abs=1e-5)
    assert np.angle(d1 / h) == pytest.approx(np.angle(complex_derivative(f, z0)), abs=1e-5)


def test_map_grid_shapes_and_images():
    grid = map_grid(np.exp, (-1, 1), (0, np.pi), n_lines=4, n_points=30)
    assert grid.horizontal.shape == grid.horizontal_image.shape == (4, 30)
    assert grid.vertical.shape == grid.vertical_image.shape == (4, 30)
    assert np.allclose(grid.horizontal.imag, grid.horizontal.imag[:, :1])
    assert np.allclose(grid.vertical.real, grid.vertical.real[:, :1])
    assert grid.vertical_image == pytest.approx(np.exp(grid.vertical))


@pytest.mark.parametrize(
    ("coefficients", "kind"),
    [
        ((1j, 0, 0, 1), "elliptic"),
        ((1, 1, 0, 1), "parabolic"),
        ((3, 0, 0, 1), "hyperbolic"),
        ((2 * np.exp(0.5j), 0, 0, 1), "loxodromic"),
        ((2, 0, 0, 2), "identity"),
    ],
)
def test_classify_mobius_kinds(coefficients, kind):
    assert classify_mobius(*coefficients).kind == kind


def test_classification_is_invariant_under_conjugation_and_fixed_points_are_fixed():
    m = np.array([[2, 1], [1, 1]], dtype=complex)  # trace^2 / det = 9: hyperbolic
    s = np.array([[1, 2j], [0.5, 3]], dtype=complex)
    conjugated = s @ m @ np.linalg.inv(s)
    first, second = classify_mobius(*m.ravel()), classify_mobius(*conjugated.ravel())
    assert first.kind == second.kind == "hyperbolic"
    assert first.trace_squared == pytest.approx(second.trace_squared)
    for p in second.fixed_points:
        assert mobius_transform(p, *conjugated.ravel()) == pytest.approx(p)


def test_affine_mobius_has_fixed_point_at_infinity():
    fixed = classify_mobius(2, 1, 0, 1).fixed_points
    assert fixed[0] == pytest.approx(-1) and np.isinf(fixed[1])
    assert np.all(np.isinf(classify_mobius(1, 1, 0, 1).fixed_points))
