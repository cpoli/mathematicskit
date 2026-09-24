"""Tests for Nesterov's accelerated gradient and Adam."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.gradient_descent import GradientDescent
from mathematicskit.optimization.systems.momentum import Adam, NesterovAcceleratedGradient

D = np.array([1.0, 1e-3])


def _f(x):
    return 0.5 * float(np.sum(D * x * x))


def _grad(x):
    return D * x


def test_nesterov_satisfies_o_one_over_k_squared_bound():
    # f(x_k) - f* <= 2 L ||x0 - x*||^2 / (k + 1)^2 with alpha = 1/L (L = 1 here).
    x0 = np.array([1.0, 1.0])
    result = NesterovAcceleratedGradient(alpha=1.0, max_iter=500).minimize(_f, _grad, x0)
    k = np.arange(len(result.path))
    fvals = np.array([_f(x) for x in result.path])
    assert np.all(fvals <= 2.0 * np.sum(x0**2) / (k + 1) ** 2 + 1e-15)


def test_nesterov_beats_gradient_descent_on_ill_conditioned_quadratic():
    x0 = np.array([1.0, 1.0])
    nag = NesterovAcceleratedGradient(alpha=1.0, max_iter=300).minimize(_f, _grad, x0)
    gd = GradientDescent(alpha=1.0, max_iter=300).minimize(_f, _grad, x0)
    assert nag.fun < 1e-2 * gd.fun


def test_nesterov_converges_to_minimizer():
    result = NesterovAcceleratedGradient(alpha=1.0, tol=1e-10, max_iter=20000).minimize(_f, _grad, np.array([3.0, -2.0]))
    assert result.converged
    np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-6)


def test_adam_first_step_moves_each_coordinate_by_alpha():
    # Bias correction makes m_hat = g and v_hat = g^2 at k = 1, so the step is alpha * sign(g).
    grad = lambda x: np.array([1e4, -1e-2])
    result = Adam(alpha=0.1, eps=1e-12, max_iter=1).minimize(lambda x: 0.0, grad, np.zeros(2))
    np.testing.assert_allclose(result.path[1], [-0.1, 0.1], rtol=1e-8)


def test_adam_is_invariant_to_objective_scaling():
    x0 = np.array([2.0, -1.0])
    a = Adam(alpha=0.01, eps=0.0, max_iter=200).minimize(_f, _grad, x0)
    b = Adam(alpha=0.01, eps=0.0, max_iter=200).minimize(lambda x: 1e6 * _f(x), lambda x: 1e6 * _grad(x), x0)
    np.testing.assert_allclose(a.path, b.path, rtol=1e-10)


def test_adam_reaches_neighbourhood_of_minimizer():
    f = lambda x: (x[0] - 1.0) ** 2 + 1000.0 * (x[1] + 2.0) ** 2
    grad = lambda x: np.array([2.0 * (x[0] - 1.0), 2000.0 * (x[1] + 2.0)])
    result = Adam(alpha=0.05, max_iter=3000).minimize(f, grad, np.zeros(2))
    np.testing.assert_allclose(result.x, [1.0, -2.0], atol=1e-2)
    assert result.fun == pytest.approx(0.0, abs=1e-3)
