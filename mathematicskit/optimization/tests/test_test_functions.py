"""Tests for the shared benchmark objective functions against their
closed-form gradients, Hessians, and known minimizers."""

import numpy as np
import pytest

from mathematicskit.optimization.utils.test_functions import (
    quadratic_bowl,
    quadratic_bowl_grad,
    quadratic_bowl_hess,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_rosenbrock_minimum_is_zero_at_the_all_ones_point(n):
    assert rosenbrock(np.ones(n)) == pytest.approx(0.0)
    assert np.allclose(rosenbrock_grad(np.ones(n)), 0.0)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_rosenbrock_gradient_matches_central_differences(n):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    h = 1e-6
    numerical = np.array([(rosenbrock(x + h * e) - rosenbrock(x - h * e)) / (2.0 * h) for e in np.eye(n)])
    assert np.allclose(rosenbrock_grad(x), numerical, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_rosenbrock_hessian_matches_central_differences_of_the_gradient(n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    h = 1e-6
    numerical = np.column_stack([(rosenbrock_grad(x + h * e) - rosenbrock_grad(x - h * e)) / (2.0 * h) for e in np.eye(n)])
    assert np.allclose(rosenbrock_hess(x), numerical, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("n", [3, 5, 10])
def test_quadratic_bowl_reports_a_dimension_mismatch_clearly(n):
    """The default bowl matrix is 2x2, so higher-dimensional x needs an
    explicit ``matrix``. Slicing ``diag([1, 10])`` to ``[:n, :n]`` silently
    left it 2x2, and the mismatch surfaced only as an opaque ``matmul``
    gufunc error from inside the objective."""
    x = np.ones(n)
    for fn in (quadratic_bowl, quadratic_bowl_grad, quadratic_bowl_hess):
        with pytest.raises(ValueError, match="2-dimensional"):
            fn(x)


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_quadratic_bowl_works_in_any_dimension_with_an_explicit_matrix(n):
    a = np.diag(np.arange(1.0, n + 1.0))
    x = np.ones(n)
    assert quadratic_bowl(x, matrix=a) == pytest.approx(0.5 * x @ a @ x)
    assert np.allclose(quadratic_bowl_grad(x, matrix=a), a @ x)
    assert np.allclose(quadratic_bowl_hess(x, matrix=a), a)


def test_quadratic_bowl_rejects_a_matrix_that_does_not_match_x():
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        quadratic_bowl(np.ones(3), matrix=np.eye(2))


def test_default_two_dimensional_bowl_is_the_documented_diag_1_10():
    assert np.allclose(quadratic_bowl_hess(np.zeros(2)), np.diag([1.0, 10.0]))
    assert quadratic_bowl(np.array([1.0, 1.0])) == pytest.approx(5.5)
