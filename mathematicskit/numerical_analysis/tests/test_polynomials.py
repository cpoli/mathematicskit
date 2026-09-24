"""Tests for Horner's scheme, Wilkinson's polynomial, and root condition numbers."""

import math

import numpy as np
import pytest

from mathematicskit.numerical_analysis.systems.polynomials import horner, root_condition_numbers, wilkinson_polynomial


def test_horner_matches_polyval_polyder_and_polydiv():
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=7)
    for x0 in (-1.3, 0.0, 0.7, 2.5):
        res = horner(coeffs, x0)
        assert res.value == pytest.approx(np.polyval(coeffs, x0), rel=1e-12, abs=1e-12)
        assert res.derivative == pytest.approx(np.polyval(np.polyder(coeffs), x0), rel=1e-12, abs=1e-12)
        quotient, remainder = np.polydiv(coeffs, [1.0, -x0])
        np.testing.assert_allclose(res.quotient, quotient, atol=1e-12)
        assert res.value == pytest.approx(remainder[-1], abs=1e-12)


def test_horner_deflation_at_a_root():
    # (x - 1)(x - 2)(x - 3) divided by (x - 2) leaves (x - 1)(x - 3).
    res = horner([1.0, -6.0, 11.0, -6.0], 2.0)
    assert res.value == 0.0
    np.testing.assert_allclose(res.quotient, [1.0, -4.0, 3.0])
    assert res.derivative == pytest.approx(-1.0)


def test_horner_constant_polynomial():
    res = horner([4.0], 10.0)
    assert res.value == 4.0 and res.derivative == 0.0 and res.quotient.shape == (0,)


def test_wilkinson_polynomial_coefficients():
    w = wilkinson_polynomial(20)
    assert w[0] == 1.0
    assert w[1] == -210.0  # -(1 + 2 + ... + 20)
    assert w[-1] == pytest.approx(math.factorial(20), rel=1e-15)
    np.testing.assert_allclose(np.sort(np.roots(wilkinson_polynomial(8)).real), np.arange(1, 9), atol=1e-8)


def test_wilkinson_perturbation_sends_roots_into_complex_plane():
    """Wilkinson (1963): a_19 -> a_19 - 2^-23 moves roots 16, 17 to about 16.73 +/- 2.81i."""
    w = wilkinson_polynomial(20)
    w[1] -= 2.0**-23
    roots = np.roots(w)
    assert np.sum(np.abs(roots.imag) > 0.5) == 10
    pair = roots[np.argmax(roots.imag)]
    assert pair.real == pytest.approx(16.7307, abs=2e-3)
    assert pair.imag == pytest.approx(2.8126, abs=2e-3)


def test_root_condition_numbers_closed_form_for_wilkinson():
    """kappa for root r w.r.t. a_19 = 210 r^18 / |w'(r)|, w'(r) = (r-1)!(20-r)!(-1)^(20-r)."""
    w = wilkinson_polynomial(20)
    for r in (1, 15, 20):
        kappa = root_condition_numbers(w, float(r))
        expected = 210.0 * r**18 / (math.factorial(r - 1) * math.factorial(20 - r))
        assert kappa[1] == pytest.approx(expected, rel=1e-3)
    assert root_condition_numbers(w, 15.0)[1] > 1e10


def test_root_condition_numbers_small_example():
    np.testing.assert_allclose(root_condition_numbers([1.0, -3.0, 2.0], 1.0), [1.0, 3.0, 2.0])


def test_root_condition_numbers_rejects_multiple_root():
    with pytest.raises(ValueError):
        root_condition_numbers([1.0, -2.0, 1.0], 1.0)
