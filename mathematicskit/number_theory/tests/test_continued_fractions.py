"""Tests for continued-fraction expansion and best rational
approximation against closed-form/known results."""

import numpy as np
import pytest

from mathematicskit.number_theory.systems.continued_fractions import best_rational_approximation, continued_fraction_expansion


def test_rational_input_terminates_exactly():
    result = continued_fraction_expansion(355 / 113, max_terms=20)
    p, q = result.convergents[-1]
    assert p == 355 and q == 113


def test_golden_ratio_is_all_ones():
    phi = (1 + 5**0.5) / 2
    result = continued_fraction_expansion(phi, max_terms=10)
    assert result.terms == [1] * 10


def test_convergents_approach_the_target_value():
    x = np.pi
    result = continued_fraction_expansion(x, max_terms=8)
    errors = [abs(p / q - x) for p, q in result.convergents]
    assert errors[-1] < errors[0]


def test_best_rational_approximation_pi():
    p, q = best_rational_approximation(np.pi, max_denominator=200)
    assert (p, q) == (355, 113)


def test_best_rational_approximation_respects_denominator_bound():
    p, q = best_rational_approximation(np.pi, max_denominator=10)
    assert q <= 10


@pytest.mark.parametrize("x", [0.5, 2.0, -3.75, 100.125])
def test_terminates_and_reconstructs_exact_rationals(x):
    result = continued_fraction_expansion(x, max_terms=30)
    p, q = result.convergents[-1]
    assert p / q == pytest.approx(x, abs=1e-9)
