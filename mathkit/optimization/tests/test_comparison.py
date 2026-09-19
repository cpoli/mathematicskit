"""Tests for the cross-method convergence-rate comparison utilities."""

import numpy as np

from mathkit.optimization.systems.gradient_descent import GradientDescent, GradientDescentLineSearch
from mathkit.optimization.systems.newton_quasi_newton import BFGS
from mathkit.optimization.utils.comparison import compare_optimizers, function_value_gap
from mathkit.optimization.utils.test_functions import quadratic_bowl, quadratic_bowl_grad


def test_compare_optimizers_returns_one_result_per_method():
    optimizers = {
        "fixed": GradientDescent(alpha=0.05, tol=1e-8),
        "line_search": GradientDescentLineSearch(tol=1e-8),
        "bfgs": BFGS(tol=1e-8),
    }
    results = compare_optimizers(optimizers, quadratic_bowl, quadratic_bowl_grad, np.array([5.0, 5.0]))
    assert set(results.keys()) == {"fixed", "line_search", "bfgs"}
    for result in results.values():
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-2)


def test_bfgs_reaches_tighter_tolerance_in_fewer_iterations_than_fixed_step_gd():
    optimizers = {
        "fixed": GradientDescent(alpha=0.09, tol=1e-8, max_iter=5000),
        "bfgs": BFGS(tol=1e-8),
    }
    results = compare_optimizers(optimizers, quadratic_bowl, quadratic_bowl_grad, np.array([5.0, 5.0]))
    assert results["bfgs"].iterations < results["fixed"].iterations


def test_function_value_gap_is_nonincreasing_for_monotone_descent_method():
    result = GradientDescent(alpha=0.05, tol=1e-10).minimize(quadratic_bowl, quadratic_bowl_grad, np.array([3.0, 3.0]))
    gap = function_value_gap(result, quadratic_bowl, f_star=0.0)
    assert np.all(np.diff(gap) <= 1e-12)
    assert gap[-1] < gap[0]
