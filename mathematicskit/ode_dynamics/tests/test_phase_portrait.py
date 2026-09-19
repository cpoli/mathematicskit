"""Tests for Linear2D/Nonlinear2D flow systems and vector-field grids."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.phase_portrait import Linear2D, Nonlinear2D, vector_field_grid


def test_linear2d_stable_node_decays_to_origin():
    system = Linear2D([1.0, 1.0], A=[[-1.0, 0.0], [0.0, -1.0]])
    result = system.integrate((0.0, 10.0), dt=1e-3, method="rk4")
    assert np.linalg.norm(result.y[-1]) < 1e-3


def test_linear2d_matches_analytic_exponential_decay():
    system = Linear2D([2.0, 0.0], A=[[-0.5, 0.0], [0.0, -0.5]])
    result = system.integrate((0.0, 4.0), dt=1e-3, method="rk4")
    expected = 2.0 * np.exp(-0.5 * result.t)
    np.testing.assert_allclose(result.y[:, 0], expected, atol=1e-4)


def test_linear2d_rejects_non_2x2_matrix():
    with pytest.raises(ValueError):
        Linear2D([1.0, 0.0], A=np.eye(3))


def test_nonlinear2d_harmonic_oscillator_conserves_amplitude():
    system = Nonlinear2D([1.0, 0.0], f=lambda x, y: (y, -x))
    result = system.integrate((0.0, 2.0 * np.pi), dt=1e-4, method="rk4")
    radius = np.sqrt(result.y[:, 0] ** 2 + result.y[:, 1] ** 2)
    assert np.max(np.abs(radius - 1.0)) < 1e-2


def test_vector_field_grid_shape_and_values():
    X, Y, U, V = vector_field_grid(lambda x, y: (y, -x), (-1, 1), (-1, 1), n=4)
    assert X.shape == Y.shape == U.shape == V.shape == (4, 4)
    # At (x, y) = (1, 0): field should be (0, -1).
    idx = np.unravel_index(np.argmin(np.abs(X - 1.0) + np.abs(Y)), X.shape)
    assert U[idx] == pytest.approx(Y[idx], abs=1e-9)
