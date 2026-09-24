"""Tests for the Frank-Wolfe method against closed-form projections and its O(1/k) gap bound."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.frank_wolfe import frank_wolfe

SIMPLEX = {"a_eq": np.ones((1, 3)), "b_eq": np.array([1.0])}


def test_projection_onto_simplex_matches_closed_form():
    # Euclidean projection of y = (1, 0.2, -0.5) onto the probability simplex
    # is (y - theta)_+ with theta = 0.1: (0.9, 0.1, 0).
    y = np.array([1.0, 0.2, -0.5])
    f = lambda x: float(np.sum((x - y) ** 2))
    grad = lambda x: 2.0 * (x - y)
    result = frank_wolfe(f, grad, [0.0, 0.0, 1.0], max_iter=3000, **SIMPLEX)
    np.testing.assert_allclose(result.x, [0.9, 0.1, 0.0], atol=2e-3)
    np.testing.assert_allclose(result.x.sum(), 1.0)


def test_gap_upper_bounds_suboptimality_and_decays():
    y = np.array([0.2, 0.3, 0.5])
    f = lambda x: float(np.sum((x - y) ** 2))
    grad = lambda x: 2.0 * (x - y)
    result = frank_wolfe(f, grad, [1.0, 0.0, 0.0], max_iter=400, **SIMPLEX)
    gaps = result.extra["gaps"]
    fvals = np.array([f(x) for x in result.path])
    assert np.all(fvals[: len(gaps)] <= gaps + 1e-12)
    # Jaggi (2013): f(x_k) - f* <= 2 C_f / (k + 2); here C_f <= diam^2 * L = 2 * 2 = 4.
    k = np.arange(len(fvals))
    assert np.all(fvals[1:] <= 8.0 / (k[1:] + 2) + 1e-12)


def test_linear_objective_stops_at_optimal_vertex():
    c = np.array([3.0, 1.0, 2.0])
    result = frank_wolfe(lambda x: float(c @ x), lambda x: c, [1 / 3, 1 / 3, 1 / 3], **SIMPLEX)
    assert result.converged
    np.testing.assert_allclose(result.x, [0.0, 1.0, 0.0], atol=1e-12)
    assert result.fun == pytest.approx(1.0)


def test_unbounded_polytope_raises():
    with pytest.raises(ValueError):
        frank_wolfe(lambda x: float(-x[0]), lambda x: np.array([-1.0]), [0.0])
