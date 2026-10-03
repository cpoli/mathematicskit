"""Tests for mathematicskit.information_theory.systems.channels."""

import numpy as np
import pytest

from mathematicskit.information_theory import (
    awgn_capacity,
    bec_capacity,
    binary_entropy,
    blahut_arimoto,
    bsc_capacity,
    bsc_transmit,
    hamming_distance,
    minimum_ebn0,
    random_code_error_rate,
    rate_distortion_binary,
    rate_distortion_gaussian,
)


def test_bsc_capacity_closed_form_and_symmetry():
    p = np.linspace(0, 1, 21)
    assert bsc_capacity(p) == pytest.approx(1 - binary_entropy(p))
    assert bsc_capacity(p) == pytest.approx(bsc_capacity(1 - p))


@pytest.mark.parametrize("p", [0.01, 0.1, 0.25, 0.4])
def test_blahut_arimoto_recovers_bsc_capacity(p):
    r = blahut_arimoto([[1 - p, p], [p, 1 - p]])
    assert r.converged
    assert r.capacity == pytest.approx(bsc_capacity(p), abs=1e-9)
    assert r.input_distribution == pytest.approx([0.5, 0.5])


def test_blahut_arimoto_erasure_and_z_channels():
    e = 0.3
    r = blahut_arimoto([[1 - e, e, 0], [0, e, 1 - e]])
    assert r.capacity == pytest.approx(bec_capacity(e), abs=1e-9)
    s = 0.2  # Z channel: 1 -> 0 with probability s; C = log2(1 + (1-s) s^(s/(1-s)))
    z = blahut_arimoto([[1, 0], [s, 1 - s]])
    assert z.capacity == pytest.approx(np.log2(1 + (1 - s) * s ** (s / (1 - s))), abs=1e-9)
    assert np.all(z.lower_bounds <= z.upper_bounds + 1e-12)
    assert z.iterations == len(z.lower_bounds)


def test_blahut_arimoto_validation_and_non_convergence():
    with pytest.raises(ValueError):
        blahut_arimoto([[0.5, 0.6], [0.5, 0.5]])
    assert not blahut_arimoto([[1, 0], [0.3, 0.7]], max_iter=1).converged


def test_awgn_capacity_and_shannon_limit():
    assert awgn_capacity(0.0) == 0.0
    assert awgn_capacity(3.0, bandwidth=2.0) == pytest.approx(4.0)
    assert minimum_ebn0(1e-10) == pytest.approx(np.log(2))
    assert minimum_ebn0(2.0) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        awgn_capacity(-1.0)
    with pytest.raises(ValueError):
        minimum_ebn0(0.0)


def test_random_codes_below_capacity_improve_with_length_and_fail_above_it():
    good = [random_code_error_rate(n, 0.25, p=0.05, trials=400, seed=n) for n in (8, 24, 40)]
    assert good[0] > good[1] > good[2]
    assert good[2] < 0.01
    bad = random_code_error_rate(40, 0.25, p=0.35, trials=200, seed=1)  # C(0.35) = 0.07 < R
    assert bad > 0.8


def test_rate_distortion_functions():
    assert rate_distortion_binary(0.0, p=0.2) == pytest.approx(binary_entropy(0.2))
    assert rate_distortion_binary(0.25, p=0.2) == 0.0
    assert rate_distortion_binary(0.1) == pytest.approx(1 - binary_entropy(0.1))
    D = np.array([0.5, 0.25, 0.125])
    assert np.diff(rate_distortion_gaussian(D)) == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        rate_distortion_gaussian(0.0)
    with pytest.raises(ValueError):
        rate_distortion_binary(-0.1)
    with pytest.raises(ValueError):
        bec_capacity(1.5)


def test_bsc_transmit_flip_rate_and_hamming_distance():
    bits = np.zeros(100_000, dtype=np.uint8)
    out = bsc_transmit(bits, 0.1, seed=0)
    assert hamming_distance(bits, out) / bits.size == pytest.approx(0.1, abs=0.005)
    with pytest.raises(ValueError):
        bsc_transmit(bits, 1.5)
    with pytest.raises(ValueError):
        hamming_distance([0, 1], [0])
