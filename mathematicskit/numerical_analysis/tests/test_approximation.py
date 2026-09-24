"""Tests for Bernstein polynomials, Padé approximants, and Remez minimax
approximation against closed-form results."""

import math

import numpy as np
import pytest

from mathematicskit.numerical_analysis.systems.approximation import PadeApproximant, bernstein_polynomial, remez_minimax


def test_bernstein_reproduces_linear_functions_exactly():
    x = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(bernstein_polynomial(lambda t: 3.0 * t - 2.0, 7, x), 3.0 * x - 2.0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 5, 40])
def test_bernstein_of_x_squared_matches_closed_form(n):
    x = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(bernstein_polynomial(lambda t: t**2, n, x), x**2 + x * (1.0 - x) / n, atol=1e-12)


def test_bernstein_converges_uniformly_for_a_kink():
    x = np.linspace(0.0, 1.0, 401)
    f = lambda t: np.abs(t - 0.5)
    errors = [np.max(np.abs(bernstein_polynomial(f, n, x) - f(x))) for n in (10, 40, 160)]
    assert errors[0] > errors[1] > errors[2]


def test_pade_2_2_of_exp_matches_closed_form():
    c = [1.0 / math.factorial(k) for k in range(5)]
    r = PadeApproximant(c, 2, 2)
    np.testing.assert_allclose(r.numerator, [1.0, 0.5, 1.0 / 12.0])
    np.testing.assert_allclose(r.denominator, [1.0, -0.5, 1.0 / 12.0])


def test_pade_matches_taylor_series_to_order_m_plus_n():
    """p - f q must vanish through x^(m+n): check the Taylor coefficients
    of p/q agree with those of log(1 + x)."""
    c = np.array([0.0] + [(-1.0) ** (k + 1) / k for k in range(1, 8)])
    r = PadeApproximant(c, 3, 3)
    # f * q truncated to degree m + n must equal p (padded with zeros).
    fq = np.convolve(c[:7], r.denominator)[:7]
    np.testing.assert_allclose(fq, np.pad(r.numerator, (0, 3)), atol=1e-12)
    assert r(0.5) == pytest.approx(np.log(1.5), abs=1e-6)


def test_pade_beats_taylor_outside_radius_of_convergence():
    c = np.array([0.0] + [(-1.0) ** (k + 1) / k for k in range(1, 9)])
    taylor = np.polynomial.polynomial.polyval(3.0, c)
    pade = PadeApproximant(c, 4, 4)(3.0)
    assert abs(pade - np.log(4.0)) < 1e-2 < abs(taylor - np.log(4.0))


def test_pade_zero_denominator_degree_is_taylor_polynomial():
    c = [1.0, 2.0, 3.0]
    r = PadeApproximant(c, 2, 0)
    np.testing.assert_allclose(r.numerator, c)
    np.testing.assert_allclose(r.denominator, [1.0])


@pytest.mark.parametrize("n", [2, 3, 5])
def test_remez_best_approximation_of_monomial(n):
    """Best degree-n approximation to x^(n+1) on [-1, 1] is x^(n+1) - T_(n+1)/2^n,
    with minimax error 2^-n (Chebyshev)."""
    res = remez_minimax(lambda x: x ** (n + 1), n)
    assert res.converged
    assert res.max_error == pytest.approx(2.0**-n, rel=1e-8)
    t = np.polynomial.chebyshev.Chebyshev.basis(n + 1).convert(kind=np.polynomial.Polynomial).coef / 2.0**n
    expected = -t[:-1]  # x^(n+1) - T_(n+1)/2^n, leading terms cancel
    np.testing.assert_allclose(res.coefficients[::-1], expected, atol=1e-9)


def test_remez_error_equioscillates_for_exp():
    n = 4
    res = remez_minimax(np.exp, n, a=0.0, b=2.0)
    assert res.converged
    err = np.exp(res.reference) - res.evaluate(res.reference)
    assert res.reference.shape == (n + 2,)
    np.testing.assert_allclose(np.abs(err), res.max_error, rtol=1e-6)
    assert np.all(np.sign(err[1:]) == -np.sign(err[:-1]))


def test_remez_beats_chebyshev_interpolation():
    f = lambda x: 1.0 / (1.0 + 25.0 * x**2)
    n = 10
    res = remez_minimax(f, n)
    nodes = np.cos(np.pi * (2 * np.arange(n + 1) + 1) / (2 * (n + 1)))
    cheb = np.polynomial.Polynomial.fit(nodes, f(nodes), n)
    x = np.linspace(-1.0, 1.0, 5001)
    assert res.max_error < np.max(np.abs(f(x) - cheb(x)))


def test_remez_linear_approximation_of_sqrt_closed_form():
    """Best line to sqrt(x) on [0, 1] is x + 1/8, error 1/8."""
    res = remez_minimax(np.sqrt, 1, a=0.0, b=1.0)
    np.testing.assert_allclose(res.coefficients, [1.0, 0.125], atol=1e-7)
    assert res.max_error == pytest.approx(0.125, abs=1e-7)
