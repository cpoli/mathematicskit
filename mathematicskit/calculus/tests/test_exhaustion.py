"""Tests for Archimedes' polygon bounds on pi."""

import math

import pytest

from mathematicskit.calculus.systems.exhaustion import archimedes_pi_bounds


def test_hexagon_start():
    result = archimedes_pi_bounds(0)
    assert result.sides == [6]
    assert result.lower == [3.0]
    assert result.upper[0] == pytest.approx(2 * math.sqrt(3))


def test_96_gon_reproduces_archimedes_bounds():
    result = archimedes_pi_bounds(4)
    assert result.sides == [6, 12, 24, 48, 96]
    assert 3 + 10 / 71 < result.lower[-1] < math.pi < result.upper[-1] < 3 + 1 / 7


def test_bounds_tighten_monotonically_and_converge():
    result = archimedes_pi_bounds(20)
    assert all(a < b for a, b in zip(result.lower, result.lower[1:]))
    assert all(a > b for a, b in zip(result.upper, result.upper[1:]))
    assert result.upper[-1] - result.lower[-1] < 1e-10


def test_gap_shrinks_by_about_a_factor_of_four_per_doubling():
    result = archimedes_pi_bounds(8)
    gaps = [u - lo for u, lo in zip(result.upper, result.lower)]
    assert gaps[-2] / gaps[-1] == pytest.approx(4.0, rel=1e-3)


def test_negative_doublings_rejected():
    with pytest.raises(ValueError):
        archimedes_pi_bounds(-1)
