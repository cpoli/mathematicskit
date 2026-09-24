"""Tests for the Weierstrass function, Koch and Hilbert curves, divider
length, and similarity dimension."""

import numpy as np
import pytest

from mathematicskit.fractals_chaos.systems.box_counting import box_counting_dimension
from mathematicskit.fractals_chaos.systems.curves import (
    divider_length,
    hilbert_curve,
    koch_curve,
    koch_snowflake,
    similarity_dimension,
    weierstrass_function,
)


def test_weierstrass_value_at_zero_is_geometric_sum():
    assert float(weierstrass_function(0.0, a=0.5, n_terms=60)) == pytest.approx(2.0)


def test_weierstrass_difference_quotients_blow_up():
    x = 0.3
    slopes = [abs(float(weierstrass_function(x + h) - weierstrass_function(x))) / h for h in (1e-2, 1e-3, 1e-4)]
    assert slopes[0] < slopes[1] < slopes[2]


@pytest.mark.parametrize("order", range(5))
def test_koch_curve_length_and_endpoints(order):
    pts = koch_curve(order)
    assert len(pts) == 4**order + 1
    assert np.allclose(pts[0], [0, 0]) and np.allclose(pts[-1], [1, 0])
    assert np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) == pytest.approx((4 / 3) ** order)


def test_koch_snowflake_is_closed_with_known_perimeter():
    pts = koch_snowflake(3)
    assert np.allclose(pts[0], pts[-1])
    assert np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) == pytest.approx(3 * (4 / 3) ** 3)


def test_koch_box_counting_dimension_near_log4_over_log3():
    assert box_counting_dimension(koch_curve(7)).dimension == pytest.approx(np.log(4) / np.log(3), abs=0.05)


@pytest.mark.parametrize("order", range(1, 6))
def test_hilbert_curve_visits_every_cell_by_unit_steps(order):
    pts = hilbert_curve(order)
    n = 2**order
    assert len({tuple(p) for p in pts}) == n * n
    assert np.all(np.abs(np.diff(pts, axis=0)).sum(axis=1) == 1)
    assert tuple(pts[0]) == (0, 0) and tuple(pts[-1]) == (n - 1, 0)


@pytest.mark.parametrize("k", range(1, 6))
def test_divider_length_on_koch_curve_is_four_thirds_power(k):
    assert divider_length(koch_curve(7), 3.0**-k) == pytest.approx((4 / 3) ** k, rel=1e-9)


def test_divider_length_of_circle_approaches_circumference():
    t = np.linspace(0, 2 * np.pi, 20001)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    assert divider_length(circle, 0.01) == pytest.approx(2 * np.pi, rel=1e-4)


@pytest.mark.parametrize(
    "ratios,expected",
    [([1 / 3, 1 / 3], np.log(2) / np.log(3)), ([0.5] * 3, np.log(3) / np.log(2)), ([1 / 3] * 8, np.log(8) / np.log(3))],
)
def test_similarity_dimension_of_classic_sets(ratios, expected):
    assert similarity_dimension(ratios) == pytest.approx(expected)


def test_similarity_dimension_rejects_non_contractions():
    with pytest.raises(ValueError):
        similarity_dimension([0.5, 1.0])
