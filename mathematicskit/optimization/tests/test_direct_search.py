"""Tests for the Nelder-Mead wrapper."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.direct_search import NelderMead
from mathematicskit.optimization.utils.test_functions import quadratic_bowl, rosenbrock


def _no_gradient(x):
    raise AssertionError("Nelder-Mead must not evaluate the gradient")


def test_minimizes_rosenbrock_without_gradients():
    result = NelderMead(tol=1e-10, max_iter=5000).minimize(rosenbrock, _no_gradient, np.array([-1.2, 1.0]))
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result.path[0], [-1.2, 1.0])
    assert len(result.path) > 10


def test_quadratic_bowl_and_nonsmooth_objective():
    result = NelderMead(tol=1e-10).minimize(quadratic_bowl, None, np.array([2.0, -1.0]))
    np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-6)
    # |x - 1| + |y + 2| is not differentiable at its minimum.
    result = NelderMead(tol=1e-10).minimize(lambda x: abs(x[0] - 1.0) + abs(x[1] + 2.0), None, np.array([0.0, 0.0]))
    np.testing.assert_allclose(result.x, [1.0, -2.0], atol=1e-6)
    assert result.fun == pytest.approx(0.0, abs=1e-6)
