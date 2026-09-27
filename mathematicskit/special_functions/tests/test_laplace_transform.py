"""Tests for the numerical Laplace transform and its Talbot and
Gaver-Stehfest inversions against closed-form transform pairs."""

import numpy as np
import pytest

from mathematicskit.special_functions.systems.laplace_transform import (
    inverse_laplace_stehfest,
    inverse_laplace_talbot,
    laplace_transform,
    stehfest_coefficients,
)

T = np.linspace(0.2, 8.0, 25)

# (f, F) closed-form pairs
PAIRS = [
    (lambda t: np.exp(-2.0 * t), lambda s: 1.0 / (s + 2.0)),
    (lambda t: t**2, lambda s: 2.0 / s**3),
    (lambda t: t * np.exp(-t), lambda s: 1.0 / (s + 1.0) ** 2),
    (lambda t: 1.0 - np.exp(-t), lambda s: 1.0 / (s * (s + 1.0))),
]


@pytest.mark.parametrize("f, F", PAIRS)
def test_forward_transform_matches_closed_form_real_s(f, F):
    s = np.array([0.5, 1.0, 3.0])
    np.testing.assert_allclose(laplace_transform(f, s), F(s), rtol=1e-8)


def test_forward_transform_of_sine_at_complex_s():
    s = np.array([1.0 + 0.5j, 2.0 - 1.0j])
    np.testing.assert_allclose(laplace_transform(np.sin, s), 1.0 / (s**2 + 1.0), rtol=1e-8)


@pytest.mark.parametrize("f, F", PAIRS)
def test_talbot_inverts_closed_form_pairs(f, F):
    np.testing.assert_allclose(inverse_laplace_talbot(F, T), f(T), atol=1e-8)


def test_talbot_inverts_oscillating_function():
    np.testing.assert_allclose(inverse_laplace_talbot(lambda s: 1.0 / (s**2 + 1.0), T), np.sin(T), atol=1e-9)


def test_talbot_needs_more_nodes_for_later_times_with_oscillating_poles():
    """Poles at -0.2 +- 1.99i leave the contour once t > m pi / (5 * 1.99)."""
    F = lambda s: (s + 0.4) / (s**2 + 0.4 * s + 4.0)
    w = np.sqrt(3.96)
    t = np.linspace(0.05, 10.0, 100)
    exact = np.exp(-0.2 * t) * (np.cos(w * t) + 0.2 / w * np.sin(w * t))
    assert np.max(np.abs(inverse_laplace_talbot(F, t, m=32) - exact)) > 1e-4
    np.testing.assert_allclose(inverse_laplace_talbot(F, t, m=40), exact, atol=1e-7)


def test_talbot_inverts_non_rational_transform():
    """L{erfc(1 / (2 sqrt t))} = exp(-sqrt s) / s."""
    from scipy.special import erfc

    f = inverse_laplace_talbot(lambda s: np.exp(-np.sqrt(s)) / s, T)
    np.testing.assert_allclose(f, erfc(1.0 / (2.0 * np.sqrt(T))), atol=1e-8)


@pytest.mark.parametrize("f, F", PAIRS)
def test_stehfest_inverts_smooth_pairs(f, F):
    np.testing.assert_allclose(inverse_laplace_stehfest(F, T), f(T), rtol=1e-3, atol=1e-3)


def test_stehfest_coefficients_sum_to_zero_and_match_known_values():
    """sum V_k = 0 because the method must map F(s) = 0 ... and exactness on f = 1 requires sum V_k / k = 1."""
    for n in (4, 8, 12, 16):
        v = stehfest_coefficients(n)
        assert v.sum() == pytest.approx(0.0, abs=1e-6 * np.abs(v).max())
        assert np.sum(v / np.arange(1, n + 1)) == pytest.approx(1.0)
    np.testing.assert_allclose(stehfest_coefficients(6), [1.0, -49.0, 366.0, -858.0, 810.0, -270.0])


def test_inversions_reject_nonpositive_times_and_odd_n():
    with pytest.raises(ValueError):
        inverse_laplace_talbot(lambda s: 1.0 / s, [0.0, 1.0])
    with pytest.raises(ValueError):
        inverse_laplace_stehfest(lambda s: 1.0 / s, [-1.0])
    with pytest.raises(ValueError):
        stehfest_coefficients(7)


def test_round_trip_forward_then_talbot():
    f = lambda t: np.exp(-t) * np.cos(2.0 * t)
    F = lambda s: (s + 1.0) / ((s + 1.0) ** 2 + 4.0)
    s = np.array([0.7, 1.9])
    np.testing.assert_allclose(laplace_transform(f, s), F(s), rtol=1e-8)
    np.testing.assert_allclose(inverse_laplace_talbot(F, T), f(T), atol=1e-8)
