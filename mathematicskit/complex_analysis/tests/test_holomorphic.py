"""Tests for complex derivatives and the Cauchy-Riemann equations."""

import numpy as np
import pytest

from mathematicskit.complex_analysis.systems.holomorphic import cauchy_riemann, complex_derivative


@pytest.mark.parametrize("f", [lambda z: z**3, np.exp, np.sin, lambda z: 1 / z])
def test_holomorphic_functions_satisfy_cauchy_riemann(f):
    assert cauchy_riemann(f, 0.4 - 0.7j).residual == pytest.approx(0.0, abs=1e-7)


def test_partials_of_z_squared():
    """z^2 = (x^2 - y^2) + 2xy i: u_x = 2x, u_y = -2y, v_x = 2y, v_y = 2x."""
    result = cauchy_riemann(lambda z: z**2, 1 + 2j)
    assert (result.u_x, result.u_y, result.v_x, result.v_y) == pytest.approx((2.0, -4.0, 4.0, 2.0))


@pytest.mark.parametrize(("f", "expected"), [(np.conj, 2.0), (np.abs, 0.8), (lambda z: z.real, 1.0)])
def test_non_holomorphic_functions_violate_cauchy_riemann(f, expected):
    """conj: u_x - v_y = 2; |z| at 0.6+0.8i: u_x = 0.6, u_y = 0.8; Re z: u_x - v_y = 1."""
    assert cauchy_riemann(f, 0.6 + 0.8j).residual == pytest.approx(expected, abs=1e-7)


def test_complex_derivative_matches_closed_form():
    z = 0.3 + 1.1j
    assert complex_derivative(np.sin, z) == pytest.approx(np.cos(z))
    assert complex_derivative(np.exp, z) == pytest.approx(np.exp(z))
    assert complex_derivative(lambda w: 1 / w, z) == pytest.approx(-1 / z**2)
