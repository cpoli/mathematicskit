"""Tests for orthogonal polynomial families against closed-form values
and their numerically-verified orthogonality."""

import numpy as np
import pytest

from mathematicskit.special_functions.systems.orthogonal_polynomials import chebyshev_polynomial, hermite_polynomial, laguerre_polynomial, legendre_polynomial
from mathematicskit.special_functions.utils.orthogonality import inner_product


def test_legendre_p2_matches_closed_form():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected = 0.5 * (3.0 * x**2 - 1.0)
    np.testing.assert_allclose(legendre_polynomial(2, x), expected)


def test_legendre_pn_at_one_is_one():
    for n in range(6):
        assert legendre_polynomial(n, 1.0) == pytest.approx(1.0)


def test_legendre_orthogonality():
    for m, n in [(2, 3), (1, 4), (0, 5)]:
        value = inner_product(lambda x, m=m: legendre_polynomial(m, x), lambda x, n=n: legendre_polynomial(n, x), lambda x: 1.0, -1.0, 1.0)
        assert abs(value) < 1e-9


def test_chebyshev_t3_matches_closed_form():
    x = np.linspace(-1, 1, 20)
    expected = 4.0 * x**3 - 3.0 * x
    np.testing.assert_allclose(chebyshev_polynomial(3, x), expected, atol=1e-10)


def test_chebyshev_matches_cosine_definition():
    theta = np.linspace(0, np.pi, 30)
    x = np.cos(theta)
    for n in range(5):
        np.testing.assert_allclose(chebyshev_polynomial(n, x), np.cos(n * theta), atol=1e-8)


def test_hermite_h2_matches_closed_form():
    x = np.linspace(-2, 2, 20)
    expected = 4.0 * x**2 - 2.0
    np.testing.assert_allclose(hermite_polynomial(2, x), expected, atol=1e-8)


def test_hermite_orthogonality_with_gaussian_weight():
    value = inner_product(lambda x: hermite_polynomial(1, x), lambda x: hermite_polynomial(2, x), lambda x: np.exp(-(x**2)), -np.inf, np.inf)
    assert abs(value) < 1e-8


def test_laguerre_l2_matches_closed_form():
    x = np.linspace(0, 5, 20)
    expected = 0.5 * (x**2 - 4.0 * x + 2.0)
    np.testing.assert_allclose(laguerre_polynomial(2, x), expected, atol=1e-8)


def test_laguerre_orthogonality_with_exponential_weight():
    value = inner_product(lambda x: laguerre_polynomial(1, x), lambda x: laguerre_polynomial(3, x), lambda x: np.exp(-x), 0.0, np.inf)
    assert abs(value) < 1e-8
