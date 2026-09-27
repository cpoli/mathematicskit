"""Tests for contours, contour integrals, winding numbers, and Cauchy's integral formula against closed-form values."""

import math

import numpy as np
import pytest

from mathematicskit.complex_analysis.systems.contours import cauchy_integral_formula, circle_contour, contour_integral, polygon_contour, winding_number

SQUARE = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]


@pytest.mark.parametrize("n", [-3, -2, 0, 1, 2, 5])
def test_integral_of_z_to_the_n_vanishes_unless_n_is_minus_one(n):
    assert contour_integral(lambda z: z**n, circle_contour(0.0, 1.5)) == pytest.approx(0j, abs=1e-10)


def test_integral_of_one_over_z_is_two_pi_i_on_circle_and_square():
    assert contour_integral(lambda z: 1 / z, circle_contour()) == pytest.approx(2j * math.pi)
    assert contour_integral(lambda z: 1 / z, polygon_contour(SQUARE)) == pytest.approx(2j * math.pi)


def test_cauchy_integral_theorem_for_entire_function_on_polygon():
    triangle = polygon_contour([0, 2 + 0.5j, -1 + 3j])
    assert contour_integral(lambda z: np.exp(z) * z**2 + np.sin(z), triangle) == pytest.approx(0j, abs=1e-10)


def test_line_integral_of_conjugate_gives_twice_the_enclosed_area():
    """oint conj(z) dz = 2i * area (Green's theorem); the square has area 4."""
    assert contour_integral(np.conj, polygon_contour(SQUARE)) == pytest.approx(8j)


def test_polygon_parametrization_hits_vertices_and_closes():
    square = polygon_contour(SQUARE)
    assert square.gamma(np.arange(5.0)) == pytest.approx(np.array(SQUARE + [SQUARE[0]]))
    assert square.points(9)[0] == square.points(9)[-1]


def test_winding_numbers():
    assert winding_number(circle_contour(), 0.2 + 0.3j) == 1
    assert winding_number(circle_contour(), 3.0) == 0
    assert winding_number(polygon_contour(SQUARE[::-1]), 0.0) == -1


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_cauchy_integral_formula_recovers_exp_and_its_derivatives(n):
    assert cauchy_integral_formula(np.exp, circle_contour(0.2, 1.0), 0.3 + 0.1j, n=n) == pytest.approx(np.exp(0.3 + 0.1j))


def test_cauchy_integral_formula_on_polygon_matches_sin_derivative():
    assert cauchy_integral_formula(np.sin, polygon_contour(SQUARE), 0.25j, n=1) == pytest.approx(np.cos(0.25j))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        circle_contour(radius=0.0)
    with pytest.raises(ValueError):
        polygon_contour([0, 1])
    with pytest.raises(ValueError):
        cauchy_integral_formula(np.exp, circle_contour(), 0.0, n=-1)
