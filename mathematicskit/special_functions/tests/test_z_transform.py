"""Tests for Z-transform evaluation, pole-zero analysis, and inversion by
partial fractions against closed-form transform pairs."""

import numpy as np
import pytest
from scipy import signal

from mathematicskit.special_functions.systems.z_transform import inverse_z_transform, poles_zeros, transfer_function, z_transform


def _impulse(n):
    x = np.zeros(n)
    x[0] = 1.0
    return x


def test_z_transform_of_geometric_sequence_approaches_closed_form():
    """sum a^n z^-n = 1 / (1 - a/z) for |z| > |a|."""
    a, z = 0.6, 1.5 * np.exp(0.4j)
    assert z_transform(a ** np.arange(200), z) == pytest.approx(1.0 / (1.0 - a / z))


def test_z_transform_of_delayed_impulse_is_z_power():
    x = np.zeros(4)
    x[3] = 1.0
    assert z_transform(x, 2.0) == pytest.approx(2.0**-3)


def test_z_transform_on_unit_circle_is_dft():
    x = np.random.default_rng(0).normal(size=8)
    z = np.exp(2j * np.pi * np.arange(8) / 8)
    np.testing.assert_allclose(z_transform(x, z), np.fft.fft(x), atol=1e-12)


def test_impulse_response_transform_equals_transfer_function():
    b, a = [1.0, 0.5], [1.0, -0.9, 0.2]
    h = signal.lfilter(b, a, _impulse(400))
    z = 1.3 * np.exp(1j * np.linspace(0.0, np.pi, 5))
    np.testing.assert_allclose(z_transform(h, z), transfer_function(b, a, z), atol=1e-10)


def test_poles_zeros_keeps_origin_zeros_and_detects_stability():
    result = poles_zeros([1.0], np.poly([0.5, -0.8]))
    np.testing.assert_allclose(sorted(result.poles.real), [-0.8, 0.5])
    np.testing.assert_allclose(result.zeros, [0.0, 0.0])
    assert result.is_stable
    assert not poles_zeros([1.0], [1.0, -1.1]).is_stable


def test_inverse_of_distinct_poles_matches_difference_equation():
    b, a = [1.0, 2.0], np.poly([0.9, -0.4, 0.3])
    np.testing.assert_allclose(inverse_z_transform(b, a, 30), signal.lfilter(b, a, _impulse(30)), atol=1e-12)


def test_inverse_of_double_pole_is_n_plus_one_times_power():
    """1 / (1 - p z^-1)^2 <-> (n + 1) p^n."""
    p = 0.7
    n = np.arange(25)
    np.testing.assert_allclose(inverse_z_transform([1.0], np.poly([p, p]), 25), (n + 1) * p**n, atol=1e-10)


def test_inverse_of_complex_pole_pair_is_damped_cosine():
    """r^n cos(w n) has transform (1 - r cos w z^-1) / (1 - 2 r cos w z^-1 + r^2 z^-2)."""
    r, w = 0.9, 0.6
    b, a = [1.0, -r * np.cos(w)], [1.0, -2 * r * np.cos(w), r**2]
    n = np.arange(40)
    h = inverse_z_transform(b, a, 40)
    assert np.isrealobj(h)
    np.testing.assert_allclose(h, r**n * np.cos(w * n), atol=1e-12)


def test_inverse_with_improper_fraction_includes_direct_terms():
    b, a = [1.0, 2.0, 3.0, 4.0], [1.0, -0.2]
    np.testing.assert_allclose(inverse_z_transform(b, a, 10), signal.lfilter(b, a, _impulse(10)), atol=1e-12)
