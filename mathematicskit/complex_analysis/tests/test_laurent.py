"""Tests for Laurent coefficients against closed-form expansions."""

import math

import numpy as np
import pytest

from mathematicskit.complex_analysis.systems.laurent import laurent_coefficients
from mathematicskit.complex_analysis.systems.residues import residue


def test_taylor_coefficients_of_exp():
    series = laurent_coefficients(np.exp, 0.0, 1.0, 10)
    for k in range(11):
        assert series.coefficient(k) == pytest.approx(1 / math.factorial(k), abs=1e-14)
    for k in range(-10, 0):
        assert series.coefficient(k) == pytest.approx(0, abs=1e-14)


def test_same_function_has_different_series_on_different_annuli():
    """1/(z(1 - z)) = 1/z + 1 + z + ... on 0 < |z| < 1, and -sum_{k>=2} z^{-k} on |z| > 1."""
    f = lambda z: 1 / (z * (1 - z))
    inner = laurent_coefficients(f, 0.0, 0.5, 5)
    outer = laurent_coefficients(f, 0.0, 2.0, 5)
    assert [inner.coefficient(k) for k in range(-2, 4)] == pytest.approx([0, 1, 1, 1, 1, 1], abs=1e-12)
    assert [outer.coefficient(k) for k in range(-5, 2)] == pytest.approx([-1, -1, -1, -1, 0, 0, 0], abs=1e-12)


def test_minus_one_coefficient_is_the_residue():
    f = lambda z: np.cos(z) / (z - 1) ** 4
    assert laurent_coefficients(f, 1.0, 0.5, 6).coefficient(-1) == pytest.approx(residue(f, 1.0))


def test_truncated_series_reproduces_function_on_annulus():
    f = lambda z: np.exp(z) / z**2
    series = laurent_coefficients(f, 0.0, 1.0, 30, n_points=128)
    z = np.array([0.4 + 0.3j, -1.2j, 1.5])
    assert series(z) == pytest.approx(f(z))


def test_invalid_order_raises():
    with pytest.raises(ValueError):
        laurent_coefficients(np.exp, 0.0, 1.0, 200, n_points=256)
