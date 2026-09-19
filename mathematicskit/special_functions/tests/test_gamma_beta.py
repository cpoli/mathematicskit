"""Tests for the gamma and beta functions against closed-form/known results."""

import math

import numpy as np
import pytest

from mathematicskit.special_functions.systems.gamma_beta import beta_function, gamma_function, log_gamma_function


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_gamma_matches_factorial_at_positive_integers(n):
    assert gamma_function(float(n)) == pytest.approx(math.factorial(n - 1))


def test_gamma_at_half_is_sqrt_pi():
    assert gamma_function(0.5) == pytest.approx(np.sqrt(np.pi))


def test_gamma_recurrence_relation():
    """Gamma(x+1) = x * Gamma(x)."""
    for x in (1.5, 2.7, 4.2):
        assert gamma_function(x + 1) == pytest.approx(x * gamma_function(x))


def test_log_gamma_matches_log_of_gamma_for_moderate_x():
    for x in (2.0, 5.0, 10.0):
        assert log_gamma_function(x) == pytest.approx(np.log(gamma_function(x)))


def test_beta_matches_gamma_ratio_definition():
    for a, b in [(2.0, 3.0), (1.5, 2.5), (4.0, 1.0)]:
        expected = gamma_function(a) * gamma_function(b) / gamma_function(a + b)
        assert beta_function(a, b) == pytest.approx(expected)


def test_beta_is_symmetric():
    assert beta_function(2.0, 5.0) == pytest.approx(beta_function(5.0, 2.0))


def test_beta_at_one_one_is_one():
    assert beta_function(1.0, 1.0) == pytest.approx(1.0)
