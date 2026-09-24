"""Tests for continued-fraction expansion and best rational
approximation against closed-form/known results."""

import math
from fractions import Fraction

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


@pytest.mark.parametrize(
    "x,max_denominator,expected",
    [
        (math.pi, 7, (22, 7)),
        (math.pi, 57, (179, 57)),  # a semiconvergent, not a convergent
        (math.pi, 99, (311, 99)),  # likewise
        (math.pi, 113, (355, 113)),
        (math.pi, 200, (355, 113)),
    ],
)
def test_best_rational_approximation_includes_semiconvergents(x, max_denominator, expected):
    """A convergent is only optimal up to *its own* denominator.

    Between two convergents the denominator can jump far past the bound --
    pi's convergents go 22/7 straight to 333/106 -- leaving room for a
    semiconvergent that is genuinely closer, such as 179/57 beating 22/7."""
    assert best_rational_approximation(x, max_denominator) == expected


@pytest.mark.parametrize("x", [math.pi, math.e, math.sqrt(2), 0.1, -math.pi, 1.0 / 3.0])
@pytest.mark.parametrize("max_denominator", [1, 2, 7, 15, 57, 100, 1000, 9999])
def test_best_rational_approximation_matches_fractions_limit_denominator(x, max_denominator):
    """Cross-check against the standard library's own optimal algorithm."""
    p, q = best_rational_approximation(x, max_denominator)
    reference = Fraction(x).limit_denominator(max_denominator)
    assert q <= max_denominator
    assert abs(p / q - x) <= abs(float(reference) - x) + 1e-15
