"""Tests for gradient descent (fixed and line-search step) against the
closed-form minimizer of a convex quadratic bowl."""

import numpy as np
import pytest

from mathkit.optimization.systems.gradient_descent import GradientDescent, GradientDescentLineSearch
from mathkit.optimization.utils.test_functions import quadratic_bowl, quadratic_bowl_grad


def test_fixed_step_converges_to_known_minimum():
    a = np.diag([2.0, 2.0])
    b = np.array([4.0, 6.0])
    f = lambda x: quadratic_bowl(x, matrix=a, b=b)
    grad = lambda x: quadratic_bowl_grad(x, matrix=a, b=b)
    result = GradientDescent(alpha=0.2, tol=1e-10, max_iter=2000).minimize(f, grad, np.array([0.0, 0.0]))
    x_star = np.linalg.solve(a, b)
    assert result.converged
    np.testing.assert_allclose(result.x, x_star, atol=1e-5)


def test_line_search_converges_faster_than_fixed_step_on_ill_conditioned_bowl():
    """The elongated default bowl (diag([1, 10])) makes fixed-step GD
    zig-zag; line search adapts and needs fewer iterations."""
    f = quadratic_bowl
    grad = quadratic_bowl_grad
    result_fixed = GradientDescent(alpha=0.09, tol=1e-8, max_iter=5000).minimize(f, grad, np.array([5.0, 5.0]))
    result_ls = GradientDescentLineSearch(tol=1e-8, max_iter=5000).minimize(f, grad, np.array([5.0, 5.0]))
    assert result_fixed.converged
    assert result_ls.converged
    assert result_ls.iterations < result_fixed.iterations


def test_path_starts_at_initial_guess():
    f = lambda x: x[0] ** 2
    grad = lambda x: np.array([2.0 * x[0]])
    result = GradientDescent(alpha=0.1).minimize(f, grad, np.array([5.0]))
    np.testing.assert_allclose(result.path[0], [5.0])


def test_gradient_norm_decreases_along_path():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    grad = lambda x: np.array([2.0 * x[0], 2.0 * x[1]])
    result = GradientDescent(alpha=0.1, max_iter=50).minimize(f, grad, np.array([3.0, 4.0]))
    grad_norms = [np.linalg.norm(grad(p)) for p in result.path]
    assert grad_norms[-1] < grad_norms[0]


def test_already_at_minimum_takes_zero_iterations():
    f = lambda x: x[0] ** 2
    grad = lambda x: np.array([2.0 * x[0]])
    result = GradientDescent(alpha=0.1, tol=1e-6).minimize(f, grad, np.array([0.0]))
    assert result.iterations == 0
    assert result.converged


@pytest.mark.parametrize("optimizer_cls", [GradientDescent, GradientDescentLineSearch])
def test_extra_dict_present(optimizer_cls):
    f = lambda x: x[0] ** 2
    grad = lambda x: np.array([2.0 * x[0]])
    kwargs = {} if optimizer_cls is GradientDescent else {}
    result = optimizer_cls(**kwargs).minimize(f, grad, np.array([2.0]))
    assert isinstance(result.method, str) and result.method
