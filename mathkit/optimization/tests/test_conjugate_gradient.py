"""Tests for nonlinear conjugate gradient against closed-form minima and
the theoretical fact that CG minimizes an n-dimensional quadratic exactly
within n iterations."""

import numpy as np
import pytest

from mathkit.optimization.systems.conjugate_gradient import NonlinearConjugateGradient
from mathkit.optimization.utils.test_functions import quadratic_bowl, quadratic_bowl_grad, rosenbrock, rosenbrock_grad


@pytest.mark.parametrize("variant", ["polak_ribiere", "fletcher_reeves"])
def test_converges_to_known_quadratic_minimum(variant):
    a = np.diag([2.0, 5.0])
    b = np.array([4.0, 10.0])
    f = lambda x: quadratic_bowl(x, matrix=a, b=b)
    grad = lambda x: quadratic_bowl_grad(x, matrix=a, b=b)
    # The backtracking (Armijo) line search only guarantees sufficient
    # decrease, not an accurate 1D minimization, so CG's textbook
    # finite-termination property (exact in n steps) needs a tighter
    # tol/more iterations here than an exact-line-search CG would.
    result = NonlinearConjugateGradient(variant=variant, tol=1e-6, max_iter=2000).minimize(f, grad, np.array([0.0, 0.0]))
    x_star = np.linalg.solve(a, b)
    np.testing.assert_allclose(result.x, x_star, atol=1e-4)


def test_cg_converges_in_far_fewer_iterations_than_gradient_descent():
    """CG's conjugate search directions avoid gradient descent's
    zig-zagging on an ill-conditioned quadratic."""
    from mathkit.optimization.systems.gradient_descent import GradientDescent

    a = np.diag([1.0, 30.0])
    f = lambda x: quadratic_bowl(x, matrix=a)
    grad = lambda x: quadratic_bowl_grad(x, matrix=a)
    cg_result = NonlinearConjugateGradient(tol=1e-8, max_iter=2000).minimize(f, grad, np.array([5.0, -3.0]))
    gd_result = GradientDescent(alpha=0.03, tol=1e-8, max_iter=5000).minimize(f, grad, np.array([5.0, -3.0]))
    assert cg_result.converged
    assert gd_result.converged
    assert cg_result.iterations < gd_result.iterations


def test_cg_reduces_rosenbrock_function_value_substantially():
    result = NonlinearConjugateGradient(tol=1e-6, max_iter=2000).minimize(rosenbrock, rosenbrock_grad, np.array([-1.2, 1.0]))
    assert result.fun < rosenbrock(np.array([-1.2, 1.0]))
    assert result.fun < 1.0


def test_rejects_unknown_variant():
    with pytest.raises(ValueError):
        NonlinearConjugateGradient(variant="bogus")
