"""Tests for Reed-Solomon encoding and erasure decoding over GF(p)."""

import itertools

import pytest

from mathematicskit.abstract_algebra.systems.codes import rs_decode_erasures, rs_encode


def test_encode_evaluates_the_message_polynomial():
    assert rs_encode([3, 1], n=5, p=7) == [3, 4, 5, 6, 0]


def test_decode_recovers_message_from_any_k_positions():
    message, n, p = [4, 0, 9, 2], 8, 13
    codeword = rs_encode(message, n, p)
    for kept in itertools.combinations(range(n), len(message)):
        received = [codeword[i] if i in kept else None for i in range(n)]
        assert rs_decode_erasures(received, k=len(message), p=p) == message


def test_decode_fails_with_too_many_erasures():
    codeword = rs_encode([1, 2, 3], n=5, p=7)
    with pytest.raises(ValueError):
        rs_decode_erasures([codeword[0], None, None, None, codeword[4]], k=3, p=7)


def test_rejects_non_prime_field_and_too_long_codewords():
    with pytest.raises(ValueError):
        rs_encode([1, 2], n=4, p=8)
    with pytest.raises(ValueError):
        rs_encode([1, 2], n=8, p=7)


def test_distinct_messages_differ_in_at_least_n_minus_k_plus_1_positions():
    n, p, k = 6, 7, 2
    words = [rs_encode(list(m), n, p) for m in itertools.product(range(p), repeat=k)]
    min_distance = min(sum(a != b for a, b in zip(u, v)) for u, v in itertools.combinations(words, 2))
    assert min_distance == n - k + 1
