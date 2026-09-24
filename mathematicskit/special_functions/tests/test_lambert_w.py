"""Tests for the Lambert W function against its defining equation and known values."""

import math

import numpy as np
import pytest

from mathematicskit.special_functions.systems.lambert_w import lambert_w


def test_principal_branch_inverts_w_exp_w():
    z = np.linspace(-1 / math.e + 1e-6, 20.0, 50)
    w = lambert_w(z)
    assert np.isrealobj(w)
    np.testing.assert_allclose(w * np.exp(w), z, rtol=1e-12, atol=1e-14)


def test_known_values():
    assert lambert_w(0.0) == pytest.approx(0.0)
    assert lambert_w(math.e) == pytest.approx(1.0)
    assert lambert_w(2 * math.e**2) == pytest.approx(2.0)


def test_omega_constant():
    """Omega = W(1) satisfies Omega = exp(-Omega) = 0.5671432904..."""
    omega = lambert_w(1.0)
    assert omega == pytest.approx(0.5671432904097838)
    assert omega == pytest.approx(math.exp(-omega))


def test_lower_branch_gives_second_real_solution():
    z = -math.log(2) / 2
    assert lambert_w(z) == pytest.approx(-math.log(2))
    assert lambert_w(z, branch=-1) == pytest.approx(-math.log(4))


def test_solves_x_to_the_x_equation():
    """x^x = y has solution x = ln y / W(ln y)."""
    y = 27.0
    x = math.log(y) / lambert_w(math.log(y))
    assert x == pytest.approx(3.0)


def test_complex_branch_returns_complex():
    w = lambert_w(1.0, branch=1)
    assert np.iscomplexobj(w)
    assert w * np.exp(w) == pytest.approx(1.0)
