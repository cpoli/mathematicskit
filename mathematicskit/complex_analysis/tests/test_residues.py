"""Tests for residues, the residue theorem, and the argument principle against closed-form values."""

import math

import numpy as np
import pytest

from mathematicskit.complex_analysis.systems.contours import circle_contour, polygon_contour
from mathematicskit.complex_analysis.systems.residues import argument_principle, residue, residue_theorem, rouche_condition


def test_simple_pole_residue():
    """Res(1/(z^2 + 1), i) = 1/(2i)."""
    assert residue(lambda z: 1 / (z**2 + 1), 1j) == pytest.approx(1 / 2j)


def test_higher_order_pole_and_essential_singularity():
    assert residue(lambda z: np.exp(z) / z**3, 0.0) == pytest.approx(0.5)
    assert residue(lambda z: np.cos(z) / (z - 1) ** 4, 1.0) == pytest.approx(math.sin(1.0) / 6)
    assert residue(lambda z: np.exp(1 / z), 0.0, radius=1.0) == pytest.approx(1.0)


def test_residue_theorem_counts_only_enclosed_poles():
    f = lambda z: 1 / ((z - 1) * (z + 2) * (z - 3j))
    result = residue_theorem(f, circle_contour(0.0, 1.5), [1, -2, 3j])
    assert result.winding_numbers.tolist() == [1, 0, 0]
    assert result.residues[0] == pytest.approx(1 / (3 * (1 - 3j)))
    assert result.integral == pytest.approx(result.predicted)
    assert result.predicted == pytest.approx(2j * math.pi * result.residues[0])


def test_residue_theorem_sum_of_residues_of_rational_function():
    """All residues of 1/(z^4 + 1) sum to zero, so a large circle gives 0."""
    poles = np.exp(1j * np.pi * (2 * np.arange(4) + 1) / 4)
    result = residue_theorem(lambda z: 1 / (z**4 + 1), circle_contour(0.0, 3.0), poles)
    assert result.integral == pytest.approx(0j, abs=1e-10)
    assert result.predicted == pytest.approx(0j, abs=1e-10)


def test_residue_evaluates_real_integral():
    """int_R dx / (1 + x^2) = 2 pi i Res(1/(1 + z^2), i) = pi."""
    assert 2j * math.pi * residue(lambda z: 1 / (1 + z**2), 1j) == pytest.approx(math.pi)


@pytest.mark.parametrize("use_derivative", [False, True])
def test_argument_principle_counts_zeros_minus_poles(use_derivative):
    f = lambda z: (z - 0.5) ** 2 * (z + 0.3j) / (z - 0.2)
    fprime = lambda z: f(z) * (2 / (z - 0.5) + 1 / (z + 0.3j) - 1 / (z - 0.2))
    kwargs = {"fprime": fprime} if use_derivative else {}
    assert argument_principle(f, circle_contour(), **kwargs) == 2
    assert argument_principle(f, circle_contour(0.5, 0.1), **kwargs) == 2
    assert argument_principle(f, polygon_contour([2, 3, 3 + 1j]), **kwargs) == 0


def test_argument_principle_counts_polynomial_roots_in_disk():
    coefficients = [1, 0, -2, 0.5, 0.1]
    inside = int(np.sum(np.abs(np.roots(coefficients)) < 1))
    assert argument_principle(lambda z: np.polyval(coefficients, z), circle_contour()) == inside


def test_rouche_condition_and_zero_counts():
    f = lambda z: z**5 + 3 * z**2 + 1
    unit, big = circle_contour(), circle_contour(0.0, 2.0)
    assert rouche_condition(f, lambda z: 3 * z**2, unit)
    assert argument_principle(f, unit) == 2
    assert rouche_condition(f, lambda z: z**5, big)
    assert argument_principle(f, big) == 5
    assert not rouche_condition(f, lambda z: z**5, unit)
