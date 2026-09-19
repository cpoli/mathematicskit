"""Tests for Chebyshev nodes/interpolation and the Runge phenomenon."""

import numpy as np

from mathkit.numerical_analysis.systems.chebyshev import ChebyshevInterpolant, chebyshev_nodes, runge_function, runge_phenomenon_errors
from mathkit.numerical_analysis.utils.error_analysis import lebesgue_constant


def test_chebyshev_nodes_span_the_interval_and_are_sorted():
    nodes = chebyshev_nodes(11, a=-2.0, b=3.0)
    assert nodes[0] == -2.0
    assert nodes[-1] == 3.0
    assert np.all(np.diff(nodes) > 0)


def test_chebyshev_interpolant_matches_smooth_function():
    p = ChebyshevInterpolant(np.sin, n=25, a=-3.0, b=3.0)
    x_test = np.linspace(-3.0, 3.0, 100)
    np.testing.assert_allclose(p.evaluate(x_test), np.sin(x_test), atol=1e-8)


def test_lebesgue_constant_smaller_for_chebyshev_than_equally_spaced():
    n = 15
    equal = np.linspace(-1.0, 1.0, n)
    cheb = chebyshev_nodes(n)
    assert lebesgue_constant(cheb) < lebesgue_constant(equal)


def test_runge_phenomenon_diverges_for_equal_nodes_converges_for_chebyshev():
    equal_err, cheb_err = runge_phenomenon_errors([5, 10, 15, 20])
    assert equal_err[-1] > equal_err[0]  # equally-spaced error grows
    assert cheb_err[-1] < cheb_err[0]  # Chebyshev error shrinks
    assert cheb_err[-1] < equal_err[-1]  # Chebyshev strictly better at high degree


def test_runge_function_value_at_zero():
    assert runge_function(0.0) == 1.0
