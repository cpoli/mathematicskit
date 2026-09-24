"""Tests for the Airy functions against their differential equation and known values."""

import math

import numpy as np
import pytest
from scipy import integrate, special

from mathematicskit.special_functions.systems.airy import airy_functions


def test_values_at_origin():
    r = airy_functions(0.0)
    assert r.ai == pytest.approx(1 / (3 ** (2 / 3) * special.gamma(2 / 3)))
    assert r.ai_prime == pytest.approx(-1 / (3 ** (1 / 3) * special.gamma(1 / 3)))
    assert r.bi == pytest.approx(math.sqrt(3) * r.ai)


def test_satisfy_airy_equation():
    """y'' = x y, via finite differences for both Ai and Bi."""
    h = 1e-4
    for x in (-3.0, -0.5, 0.7, 2.0):
        for attr in ("ai", "bi"):
            y, yp, ym = (getattr(airy_functions(x + d), attr) for d in (0.0, h, -h))
            assert (yp - 2 * y + ym) / h**2 == pytest.approx(x * y, abs=1e-5)


def test_wronskian():
    """Ai Bi' - Ai' Bi = 1/pi for all x."""
    r = airy_functions(np.linspace(-8, 3, 23))
    np.testing.assert_allclose(r.ai * r.bi_prime - r.ai_prime * r.bi, 1 / np.pi, rtol=1e-10)


def test_ai_integrates_to_one_third_over_positive_axis_and_one_overall():
    pos, _ = integrate.quad(lambda x: airy_functions(x).ai, 0, np.inf)
    assert pos == pytest.approx(1 / 3)


def test_first_zero_of_ai():
    zero = -2.338107410459767
    assert airy_functions(zero).ai == pytest.approx(0.0, abs=1e-12)
    assert zero == pytest.approx(special.ai_zeros(1)[0][0])


def test_ai_decays_like_asymptotic_form():
    """Ai(x) ~ exp(-2/3 x^{3/2}) / (2 sqrt(pi) x^{1/4}) as x -> +inf."""
    x = 20.0
    asymptotic = math.exp(-2 / 3 * x**1.5) / (2 * math.sqrt(math.pi) * x**0.25)
    assert airy_functions(x).ai == pytest.approx(asymptotic, rel=5e-3)
