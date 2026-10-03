"""Tests for the primal-dual interior-point LP solver."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.interior_point import interior_point_lp
from mathematicskit.optimization.systems.linear_programming import linear_program


def test_production_lp_vertex_optimum():
    c, a, b = np.array([-3.0, -5.0]), np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([40.0, 30.0])
    result = interior_point_lp(c, a, b)
    assert result.converged and result.method == "interior_point"
    assert np.allclose(result.x, [20 / 3, 50 / 3], atol=1e-7)
    # Strong duality: b^T y = c^T x, with y <= 0 for the <= constraints.
    y = result.extra["dual"]
    assert b @ y == pytest.approx(result.fun, rel=1e-7)
    assert np.all(y <= 1e-9)


def test_agrees_with_highs_ipm_and_simplex_on_random_lps():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.uniform(0.1, 1.0, size=(6, 4))
        b = rng.uniform(1.0, 2.0, size=6)
        c = -rng.uniform(0.5, 1.5, size=4)
        ours = interior_point_lp(c, a, b)
        ipm = linear_program(c, a_ub=a, b_ub=b, method="highs-ipm")
        simplex = linear_program(c, a_ub=a, b_ub=b, method="highs-ds")
        assert ours.converged and ipm.success and simplex.success
        assert ours.fun == pytest.approx(ipm.fun, rel=1e-7) and ours.fun == pytest.approx(simplex.fun, rel=1e-7)


def test_iterates_stay_interior_and_gap_shrinks():
    c, a, b = np.array([-1.0, -1.0]), np.array([[1.0, 3.0], [3.0, 1.0]]), np.array([6.0, 6.0])
    result = interior_point_lp(c, a, b)
    path = result.path
    assert np.all(path > 0)
    feasible = path[np.all(path @ a.T <= b + 1e-9, axis=1)]
    assert np.all(feasible @ a.T < b + 1e-9)
    mu = result.extra["duality_measure"]
    assert mu[-1] < 1e-9 < mu[0]
    assert np.all(np.diff(mu) < 0)


def test_max_iter_reports_nonconvergence():
    result = interior_point_lp(np.array([-1.0]), np.array([[1.0]]), np.array([1.0]), max_iter=2)
    assert not result.converged and result.path.shape == (3, 1)
