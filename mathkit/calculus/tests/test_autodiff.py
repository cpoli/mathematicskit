"""Tests for reverse-mode (backpropagation-style) automatic differentiation
against closed-form gradients."""

import pytest

from mathkit.calculus.systems.autodiff import Variable, gradient


def test_single_variable_matches_closed_form_derivative():
    f = lambda x: x * x * x - 2.0 * x  # f' = 3x^2 - 2
    x0 = 2.0
    (grad,) = gradient(f, [x0])
    assert grad == pytest.approx(3.0 * x0**2 - 2.0, abs=1e-10)


def test_multivariable_gradient_matches_closed_form():
    f = lambda x, y: x * x * y + y  # df/dx = 2xy, df/dy = x^2 + 1
    x0, y0 = 3.0, 2.0
    grad = gradient(f, [x0, y0])
    assert grad[0] == pytest.approx(2.0 * x0 * y0, abs=1e-10)
    assert grad[1] == pytest.approx(x0**2 + 1.0, abs=1e-10)


def test_shared_subexpression_gradient_is_correct():
    """x used twice (f = x*x) must accumulate both contributions to grad,
    not overwrite -- the key correctness property of reverse-mode
    autodiff's graph accumulation."""
    x = Variable(4.0)
    y = x * x
    y.backward()
    assert x.grad == pytest.approx(8.0)  # d(x^2)/dx = 2x = 8


def test_trig_and_exp_gradients():
    import math

    f = lambda x: x.sin() * x.exp()
    x0 = 0.5
    (grad,) = gradient(f, [x0])
    expected = math.cos(x0) * math.exp(x0) + math.sin(x0) * math.exp(x0)
    assert grad == pytest.approx(expected, rel=1e-10)


def test_repeated_backward_calls_reset_gradients():
    x = Variable(2.0)
    y = x * x
    y.backward()
    grad1 = x.grad
    y.backward()
    grad2 = x.grad
    assert grad1 == pytest.approx(grad2)
