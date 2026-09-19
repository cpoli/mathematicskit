"""Tests for Maclaurin series expansion and convergence-radius estimation."""

import numpy as np
import pytest

from mathkit.calculus.systems.taylor_series import estimate_radius_of_convergence, evaluate_series, maclaurin_coefficients, taylor_remainder_bound


@pytest.mark.parametrize(
    "name,func,x",
    [
        ("exp", np.exp, 1.0),
        ("sin", np.sin, 0.5),
        ("cos", np.cos, 0.5),
        ("log1p", np.log1p, 0.3),
        ("arctan", np.arctan, 0.4),
    ],
)
def test_series_converges_to_known_function_value(name, func, x):
    coeffs = maclaurin_coefficients(name, order=30)
    approx = evaluate_series(coeffs, x)
    assert approx == pytest.approx(func(x), abs=1e-8)


def test_geometric_series_matches_closed_form():
    coeffs = maclaurin_coefficients("geometric", order=40)
    x = 0.5
    approx = evaluate_series(coeffs, x)
    assert approx == pytest.approx(1.0 / (1.0 - x), abs=1e-8)


def test_radius_of_convergence_for_entire_functions_grows_with_order():
    """exp/sin/cos have infinite radius of convergence; the ratio-test
    estimate from a finite truncation should grow (not converge to a
    finite bound) as more terms are included."""
    small = estimate_radius_of_convergence(maclaurin_coefficients("exp", order=10))
    large = estimate_radius_of_convergence(maclaurin_coefficients("exp", order=30))
    assert large > small


def test_radius_of_convergence_for_log1p_and_geometric_is_one():
    assert estimate_radius_of_convergence(maclaurin_coefficients("geometric", order=50)) == pytest.approx(1.0, abs=1e-6)
    assert estimate_radius_of_convergence(maclaurin_coefficients("log1p", order=500)) == pytest.approx(1.0, abs=1e-2)


def test_taylor_remainder_bound_shrinks_with_order():
    bounds = [taylor_remainder_bound(1.0, order, 0.5) for order in (2, 5, 10, 15)]
    assert bounds == sorted(bounds, reverse=True)


def test_unknown_series_name_rejected():
    with pytest.raises(ValueError):
        maclaurin_coefficients("not_a_function", order=5)
