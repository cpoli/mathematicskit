"""Tests for mathematicskit.topology.systems.fixed_points and curves."""

import numpy as np
import pytest
from scipy.optimize import fsolve

from mathematicskit.topology import (
    brouwer_fixed_point,
    circle,
    fully_labeled_triangles,
    is_sperner_labeling,
    lefschetz_number,
    linking_number,
    random_sperner_labeling,
    sphere,
    torus,
    triangle_grid,
    turning_number,
    vector_field_index,
)


@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_triangle_grid_has_n_squared_triangles(n):
    grid = triangle_grid(n)
    assert grid.triangles.shape == (n * n, 3)
    assert np.all(grid.points.sum(axis=1) == n)
    assert np.allclose(grid.cartesian[grid.points[:, 2] == n], [[0.5, np.sqrt(3) / 2]])


@pytest.mark.parametrize("seed", range(10))
def test_sperner_lemma_fully_labeled_count_is_odd(seed):
    grid = triangle_grid(15)
    labels = random_sperner_labeling(grid, seed=seed)
    assert is_sperner_labeling(grid, labels)
    assert len(fully_labeled_triangles(grid, labels)) % 2 == 1


def test_non_sperner_labeling_is_detected():
    grid = triangle_grid(3)
    labels = np.zeros(len(grid.points), dtype=int)
    assert not is_sperner_labeling(grid, labels)


def test_brouwer_fixed_point_converges_to_the_true_fixed_point():
    def f(x):
        y = np.array([x[1] ** 2 + 0.2, np.sin(x[0]) * 0.5 + 0.1, 0.0])
        y[2] = max(0.0, 1 - y[0] - y[1])
        return y / y.sum()

    exact = fsolve(lambda z: f(np.array([z[0], z[1], 1 - z[0] - z[1]]))[:2] - z, [0.3, 0.3])
    errors = [np.max(np.abs(brouwer_fixed_point(f, n).point[:2] - exact)) for n in (10, 40, 160)]
    assert errors[-1] < 0.01 and errors[-1] < errors[0]
    assert brouwer_fixed_point(f, 160).residual < 0.02


@pytest.mark.parametrize(
    ("field", "index"),
    [
        (lambda x, y: (x, y), 1),
        (lambda x, y: (-y, x), 1),
        (lambda x, y: (x, -y), -1),
        (lambda x, y: (x**3 - 3 * x * y**2, 3 * x**2 * y - y**3), 3),
        (lambda x, y: (x**2 - y**2, -2 * x * y), -2),
        (lambda x, y: (1.0, 0.0), 0),
    ],
)
def test_vector_field_index_of_standard_zeros(field, index):
    assert vector_field_index(field, radius=0.5) == index


def test_poincare_hopf_outward_field_on_disk_has_total_index_one():
    def gradient(x, y):  # two sources and a saddle, pointing outward on the unit circle
        return (x**3 - 0.25 * x, y)

    assert vector_field_index(gradient, radius=1.0) == 1
    assert vector_field_index(gradient, center=(0.5, 0.0), radius=0.2) == 1
    assert vector_field_index(gradient, center=(0.0, 0.0), radius=0.2) == -1
    with pytest.raises(ValueError):
        vector_field_index(lambda x, y: (x, y), center=(1.0, 0.0), radius=1.0)


def test_lefschetz_number_identity_is_euler_characteristic_and_antipodal_map():
    for K in (circle(5), sphere(2), torus(3, 3)):
        assert lefschetz_number(K, list(range(len(K.vertices)))) == K.euler_characteristic
    octahedron = sphere(2, "cross_polytope")
    assert lefschetz_number(octahedron, [v ^ 1 for v in range(6)]) == 0
    assert lefschetz_number(circle(5), [(-v) % 5 for v in range(5)]) == 2
    assert lefschetz_number(circle(6), {v: 0 for v in range(6)}) == 1
    with pytest.raises(ValueError):
        lefschetz_number(circle(6), [(2 * v) % 6 for v in range(6)])


def _ring(t, center=(0, 0, 0), plane="xy"):
    c, s = np.cos(t), np.sin(t)
    z = 0 * t
    pts = {"xy": (c, s, z), "xz": (c, z, s)}[plane]
    return np.column_stack(pts) + center


def test_linking_number_of_hopf_link_unlink_and_torus_link():
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    a = _ring(t)
    assert linking_number(a, _ring(t, (1, 0, 0), "xz")) == pytest.approx(-1, abs=1e-6)
    assert linking_number(a, _ring(t, (3, 0, 0), "xz")) == pytest.approx(0, abs=1e-6)

    # the (2, 4) torus link: two parallel curves winding once around and twice through the torus
    def torus_curve(phase):
        v = 2 * t + phase
        return np.column_stack([(2 + np.cos(v)) * np.cos(t), (2 + np.cos(v)) * np.sin(t), np.sin(v)])

    assert abs(linking_number(torus_curve(0.0), torus_curve(np.pi))) == pytest.approx(2, abs=1e-6)


def test_linking_number_is_symmetric():
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    a, b = _ring(t), _ring(t, (1, 0, 0), "xz")
    assert linking_number(a, b) == pytest.approx(linking_number(b, a))


def test_turning_numbers():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    assert turning_number(np.column_stack([np.cos(t), np.sin(t)])) == 1
    assert turning_number(np.column_stack([np.cos(t), -np.sin(t)])) == -1
    assert turning_number(np.column_stack([np.sin(t), np.sin(2 * t)])) == 0
    limacon = (0.5 + np.cos(t))[:, None] * np.column_stack([np.cos(t), np.sin(t)])
    assert turning_number(limacon) == 2
    star = np.array([[np.cos(4 * np.pi * k / 5), np.sin(4 * np.pi * k / 5)] for k in range(5)])
    assert turning_number(star) == 2
