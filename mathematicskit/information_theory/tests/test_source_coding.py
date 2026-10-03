"""Tests for mathematicskit.information_theory.systems.source_coding."""

import itertools
import math

import numpy as np
import pytest

from mathematicskit.information_theory import (
    PrefixCode,
    arithmetic_decode,
    arithmetic_encode,
    arithmetic_intervals,
    canonical_code,
    elias_gamma_decode,
    elias_gamma_encode,
    entropy,
    huffman_code,
    kraft_sum,
    lz78_compressed_bits,
    lz78_decode,
    lz78_encode,
    shannon_fano_code,
    typical_set,
)

RNG = np.random.default_rng(0)
DISTRIBUTIONS = [RNG.dirichlet(np.ones(k)) for k in (2, 3, 5, 8, 13)]


@pytest.mark.parametrize("p", DISTRIBUTIONS)
def test_huffman_is_within_one_bit_of_entropy_and_beats_shannon_fano(p):
    code = huffman_code(p)
    assert code.entropy <= code.average_length + 1e-12 < code.entropy + 1
    assert kraft_sum([len(w) for w in code.codewords.values()]) == pytest.approx(1.0)
    for method in ("fano", "shannon"):
        sf = shannon_fano_code(p, method=method)
        assert sf.entropy <= sf.average_length < sf.entropy + 1
        assert code.average_length <= sf.average_length + 1e-12


def test_huffman_is_optimal_by_brute_force_over_kraft_feasible_lengths():
    p = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    best = min(sum(p * l) for l in itertools.product(range(1, 5), repeat=len(p)) if kraft_sum(l) <= 1)
    assert huffman_code(p).average_length == pytest.approx(best)


def test_dyadic_source_is_coded_at_exactly_its_entropy():
    code = huffman_code({"a": 0.5, "b": 0.25, "c": 0.125, "d": 0.125})
    assert code.average_length == pytest.approx(code.entropy) == pytest.approx(1.75)
    assert code.efficiency == pytest.approx(1.0)


def test_shannon_code_lengths_are_ceil_of_self_information():
    p = [0.4, 0.3, 0.2, 0.1]
    code = shannon_fano_code(p, method="shannon")
    assert [len(code.codewords[i]) for i in range(4)] == [math.ceil(-math.log2(x)) for x in p]
    with pytest.raises(ValueError):
        shannon_fano_code(p, method="morse")


def test_prefix_code_round_trip_and_validation():
    code = huffman_code({"x": 0.6, "y": 0.3, "z": 0.1})
    message = list("xyzzyxxx")
    assert code.decode(code.encode(message)) == message
    assert huffman_code({"only": 1.0}).codewords == {"only": "0"}
    with pytest.raises(ValueError):
        PrefixCode({"a": "0", "b": "01"}, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):
        code.decode("0")  # a proper prefix of two codewords


def test_kraft_sum_and_canonical_code():
    lengths = [3, 1, 3, 2]
    words = canonical_code(lengths)
    assert [len(w) for w in words] == lengths
    PrefixCode(dict(enumerate(words)), {})  # raises if not prefix-free
    assert kraft_sum([1, 1], radix=3) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        canonical_code([1, 1, 1])


def test_typical_set_matches_brute_force_enumeration():
    p, n, eps = np.array([0.7, 0.3]), 12, 0.15
    H = entropy(p)
    size, prob = 0, 0.0
    for x in itertools.product((0, 1), repeat=n):
        px = np.prod(p[list(x)])
        if abs(-np.log2(px) / n - H) <= eps:
            size += 1
            prob += px
    r = typical_set(p, n, eps)
    assert 2**r.log2_size == pytest.approx(size)
    assert r.probability == pytest.approx(prob)


def test_asymptotic_equipartition_property():
    p, eps = [0.8, 0.15, 0.05], 0.1
    probs = [typical_set(p, n, eps).probability for n in (50, 200, 800)]
    assert probs[0] < probs[1] < probs[2] and probs[2] > 0.95
    r = typical_set(p, 800, eps)
    assert r.n * (r.entropy - eps) + np.log2(1 - eps) <= r.log2_size <= r.n * (r.entropy + eps)
    with pytest.raises(ValueError):
        typical_set([1.0, 0.0], 5, 0.1)


def test_elias_gamma_round_trip_and_length():
    numbers = list(range(1, 300))
    bits = "".join(elias_gamma_encode(n) for n in numbers)
    assert elias_gamma_decode(bits) == numbers
    assert all(len(elias_gamma_encode(n)) == 2 * int(math.log2(n)) + 1 for n in numbers)
    with pytest.raises(ValueError):
        elias_gamma_encode(0)
    with pytest.raises(ValueError):
        elias_gamma_decode("001")


def test_arithmetic_coding_round_trip_and_length_bound():
    p = {"a": 0.6, "b": 0.3, "c": 0.1}
    rng = np.random.default_rng(1)
    for _ in range(20):
        message = list(rng.choice(list(p), size=30, p=list(p.values())))
        bits = arithmetic_encode(message, p)
        assert arithmetic_decode(bits, p, len(message)) == message
        self_information = -sum(math.log2(p[s]) for s in message)
        assert len(bits) <= math.ceil(self_information) + 1


def test_arithmetic_interval_width_is_message_probability():
    low, high = arithmetic_intervals("abca", {"a": 0.5, "b": 0.25, "c": 0.25})[-1]
    assert float(high - low) == pytest.approx(0.5 * 0.25 * 0.25 * 0.5)
    with pytest.raises(ValueError):
        arithmetic_encode("ab", {"a": 1.0, "b": 0.0})


def test_lz78_round_trip_and_compression_of_low_entropy_source():
    rng = np.random.default_rng(2)
    seq = (rng.random(20000) < 0.05).astype(int).tolist()
    pairs = lz78_encode(seq)
    assert lz78_decode(pairs) == seq
    assert lz78_compressed_bits(pairs, 2) / len(seq) < 0.6
    assert lz78_decode(lz78_encode("ABA")) == list("ABA")
