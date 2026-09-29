"""Tests for floating-point arithmetic helpers against numpy.finfo, the
IEEE 754 layouts, and closed forms (Higham, Accuracy and Stability of
Numerical Algorithms, 2nd ed., Ch. 1-2)."""

import math
import struct

import numpy as np
import pytest

from mathematicskit.numerical_analysis import cancellation_bits_lost, float_bits, machine_epsilon, quadratic_roots, toy_float_system, ulp


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_machine_epsilon_matches_finfo(dtype):
    assert machine_epsilon(dtype) == float(np.finfo(dtype).eps)


def test_machine_epsilon_rejects_non_float_dtype():
    with pytest.raises(ValueError):
        machine_epsilon(np.int64)


@pytest.mark.parametrize("x", [1.0, -2.5, 0.1, 1e300, 1e-310, 5e-324, 0.0, -0.0, math.pi])
def test_float_bits_reassembles_binary64_value(x):
    bits = float_bits(x)
    assert bits.format == "binary64"
    assert len(bits.exponent_bits) == 11 and len(bits.fraction_bits) == 52
    assert (-1) ** bits.sign * bits.significand * 2.0**bits.exponent == x
    word = bits.sign * 2**63 + int(bits.exponent_bits + bits.fraction_bits, 2)
    assert word == struct.unpack(">Q", struct.pack(">d", x))[0]


def test_float_bits_categories():
    assert float_bits(0.0).category == "zero"
    assert float_bits(-0.0).sign == 1
    assert float_bits(1e-310).category == "subnormal"
    assert float_bits(1.0).category == "normal"
    assert float_bits(math.inf).category == "infinity"
    nan = float_bits(math.nan)
    assert nan.category == "nan"
    assert math.isnan(nan.significand)


@pytest.mark.parametrize("dtype,e_bits,f_bits", [(np.float16, 5, 10), (np.float32, 8, 23)])
def test_float_bits_narrow_formats(dtype, e_bits, f_bits):
    bits = float_bits(0.1, dtype)
    assert (len(bits.exponent_bits), len(bits.fraction_bits)) == (e_bits, f_bits)
    assert bits.value == float(dtype(0.1))
    assert bits.significand * 2.0**bits.exponent == bits.value
    # Largest finite value: all-ones exponent field minus one, full fraction.
    top = float_bits(float(np.finfo(dtype).max), dtype)
    assert top.exponent_bits == "1" * (e_bits - 1) + "0"
    assert top.fraction_bits == "1" * f_bits


def test_ulp_doubles_at_each_power_of_two_and_matches_spacing():
    np.testing.assert_array_equal(ulp([1.0, 1.5, 2.0, 3.99, 4.0]), [2.0**-52, 2.0**-52, 2.0**-51, 2.0**-51, 2.0**-50])
    assert ulp(-8.0) == ulp(8.0)
    assert ulp(1.0, np.float32) == 2.0**-23
    assert 1.0 + ulp(1.0) / 2 == 1.0  # ties round to even


def test_toy_float_system_matches_higham_figure():
    """Higham's Figure 2.1 system (t = 3, emin = -1, emax = 3): 4 numbers
    per binade from 0.25 to 7, with spacing 2**(e - 3)."""
    values = toy_float_system()
    assert values.size == 4 * 5
    assert values[0] == 0.25 and values[-1] == 7.0
    gaps = np.diff(values)
    np.testing.assert_allclose(gaps[values[:-1] >= 4.0], 1.0)
    np.testing.assert_allclose(gaps[(values[:-1] >= 1.0) & (values[:-1] < 2.0)], 0.25)
    # The gap after 1 is the system's machine epsilon, 2**(1 - t).
    assert values[values > 1.0][0] - 1.0 == 2.0 ** (1 - 3)


def test_toy_float_system_subnormals_fill_the_gap_evenly():
    values = toy_float_system(precision=3, emin=-1, emax=0, subnormals=True)
    below = values[values < 0.25]
    np.testing.assert_allclose(below, [1 / 16, 2 / 16, 3 / 16])
    with pytest.raises(ValueError):
        toy_float_system(precision=0)


def test_quadratic_roots_agree_when_there_is_no_cancellation():
    for stable in (True, False):
        np.testing.assert_allclose(quadratic_roots(2.0, -3.0, -2.0, stable=stable), [-0.5, 2.0])
    np.testing.assert_array_equal(quadratic_roots(1.0, 0.0, 0.0), [0.0, 0.0])


@pytest.mark.parametrize("b", [1e8, -1e8])
def test_stable_quadratic_formula_avoids_cancellation(b):
    small_exact = -1.0 / b  # product of roots is 1, and the big root is about -b
    naive = quadratic_roots(1.0, b, 1.0, stable=False)
    stable = quadratic_roots(1.0, b, 1.0)
    small_stable = stable[np.argmin(np.abs(stable))]
    small_naive = naive[np.argmin(np.abs(naive))]
    assert small_stable == pytest.approx(small_exact, rel=1e-15)
    assert abs(small_naive - small_exact) / abs(small_exact) > 0.1


def test_quadratic_roots_rejects_degenerate_input():
    with pytest.raises(ValueError):
        quadratic_roots(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        quadratic_roots(1.0, 0.0, 1.0)


def test_cancellation_bits_lost_bounds_the_observed_loss():
    """Loss-of-precision theorem: 1 - cos(x) for small x loses about
    -log2(x**2 / 2) bits, so cos(x)'s rounding error (at most u = 2**-53)
    is amplified by up to 2**lost in the difference."""
    x = 1e-5
    lost = cancellation_bits_lost(1.0, math.cos(x))
    naive, exact = 1.0 - math.cos(x), 2.0 * math.sin(x / 2) ** 2
    rel_err = abs(naive - exact) / exact
    assert lost == pytest.approx(-math.log2(x * x / 2), abs=1e-3)
    assert 1e3 * 2.0**-53 < rel_err <= 2.0 ** (lost - 53)
    assert cancellation_bits_lost(1.0, 0.5) == 1.0
    assert cancellation_bits_lost(1.0, -1.0) == 0.0
    assert cancellation_bits_lost(3.0, 3.0) == math.inf
    with pytest.raises(ValueError):
        cancellation_bits_lost(0.0, 1.0)
