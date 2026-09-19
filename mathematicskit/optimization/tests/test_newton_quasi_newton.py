"""Tests for Newton's method and BFGS against closed-form minima,
including the Rosenbrock function's known global minimum."""

import numpy as np

from mathematicskit.optimization.systems.newton_quasi_newton import BFGS, NewtonMethod
from mathematicskit.optimization.utils.test_functions import (
    quadratic_bowl,
    quadratic_bowl_grad,
    quadratic_bowl_hess,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)


def test_newton_converges_to_rosenbrock_minimum():
    result = NewtonMethod(tol=1e-10).minimize(rosenbrock, rosenbrock_grad, np.array([-1.2, 1.0]), hess=rosenbrock_hess)
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)
    assert result.fun < 1e-6


def test_bfgs_converges_to_rosenbrock_minimum():
    result = BFGS(tol=1e-10).minimize(rosenbrock, rosenbrock_grad, np.array([-1.2, 1.0]))
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-3)


def test_newton_converges_to_known_quadratic_minimum():
    a = np.diag([2.0, 5.0])
    b = np.array([4.0, 10.0])
    f = lambda x: quadratic_bowl(x, matrix=a, b=b)
    grad = lambda x: quadratic_bowl_grad(x, matrix=a, b=b)
    hess = lambda x: quadratic_bowl_hess(x, matrix=a, b=b)
    result = NewtonMethod(tol=1e-10).minimize(f, grad, np.array([0.0, 0.0]), hess=hess)
    x_star = np.linalg.solve(a, b)
    np.testing.assert_allclose(result.x, x_star, atol=1e-6)


def test_newton_path_records_every_scipy_iterate():
    result = NewtonMethod().minimize(rosenbrock, rosenbrock_grad, np.array([-1.2, 1.0]), hess=rosenbrock_hess)
    assert result.path.shape[0] == result.iterations + 1
    np.testing.assert_allclose(result.path[0], [-1.2, 1.0])


def test_newton_faster_than_bfgs_in_iteration_count_is_not_assumed_but_both_converge():
    """Newton and BFGS both solve the quadratic bowl exactly/near-exactly
    since it is convex quadratic (Newton in one step)."""
    f, grad, hess = quadratic_bowl, quadratic_bowl_grad, quadratic_bowl_hess
    newton_result = NewtonMethod(tol=1e-12).minimize(f, grad, np.array([5.0, 5.0]), hess=hess)
    bfgs_result = BFGS(tol=1e-10).minimize(f, grad, np.array([5.0, 5.0]))
    np.testing.assert_allclose(newton_result.x, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(bfgs_result.x, [0.0, 0.0], atol=1e-4)
