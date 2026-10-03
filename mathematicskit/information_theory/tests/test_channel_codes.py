"""Tests for mathematicskit.information_theory.systems.channel_codes."""

import itertools

import numpy as np
import pytest

from mathematicskit.information_theory import (
    bit_flip_decode,
    bsc_transmit,
    convolutional_encode,
    gallager_ldpc_matrix,
    hamming_decode,
    hamming_encode,
    hamming_parity_check_matrix,
    viterbi_decode,
)


@pytest.mark.parametrize("r", [2, 3, 4])
def test_hamming_codewords_satisfy_every_check_and_correct_every_single_error(r):
    n, k = 2**r - 1, 2**r - r - 1
    H = hamming_parity_check_matrix(r)
    for message in itertools.islice(itertools.product((0, 1), repeat=k), 64):
        word = hamming_encode(message, r)
        assert not ((H.astype(int) @ word) % 2).any()
        assert hamming_decode(word, r).tolist() == list(message)
        for i in range(n):
            corrupted = word.copy()
            corrupted[i] ^= 1
            assert hamming_decode(corrupted, r).tolist() == list(message)


def test_hamming_7_4_code_has_minimum_distance_3():
    weights = [hamming_encode(m).sum() for m in itertools.product((0, 1), repeat=4) if any(m)]
    assert min(weights) == 3
    assert sorted(weights).count(3) == 7


def test_hamming_multiple_blocks_and_validation():
    message = np.random.default_rng(0).integers(0, 2, 40)
    assert hamming_decode(hamming_encode(message)).tolist() == message.tolist()
    with pytest.raises(ValueError):
        hamming_encode([1, 0, 1])
    with pytest.raises(ValueError):
        hamming_decode([1, 0, 1])
    with pytest.raises(ValueError):
        hamming_parity_check_matrix(1)
    with pytest.raises(ValueError):
        hamming_encode([2, 0, 0, 0])


def test_convolutional_code_is_linear_with_free_distance_5():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 2, 20), rng.integers(0, 2, 20)
    assert (convolutional_encode(a) ^ convolutional_encode(b)).tolist() == convolutional_encode(a ^ b).tolist()
    weights = [convolutional_encode(m).sum() for L in range(1, 8) for m in itertools.product((0, 1), repeat=L) if m[0]]
    assert min(weights) == 5


def test_viterbi_corrects_any_two_errors_in_a_short_block():
    message = [1, 0, 1, 1, 0, 0, 1]
    word = convolutional_encode(message)
    assert viterbi_decode(word).tolist() == message
    for i, j in itertools.combinations(range(word.size), 2):
        corrupted = word.copy()
        corrupted[[i, j]] ^= 1
        assert viterbi_decode(corrupted).tolist() == message


def test_viterbi_beats_uncoded_transmission_on_a_bsc():
    rng = np.random.default_rng(2)
    message = rng.integers(0, 2, 5000)
    decoded = viterbi_decode(bsc_transmit(convolutional_encode(message), 0.03, seed=3))
    uncoded = bsc_transmit(message, 0.03, seed=4)
    assert np.mean(decoded != message) < 0.25 * np.mean(uncoded != message)
    with pytest.raises(ValueError):
        viterbi_decode([1, 0, 1])


def test_gallager_matrix_is_regular():
    H = gallager_ldpc_matrix(60, column_weight=3, row_weight=6, seed=0)
    assert H.shape == (30, 60)
    assert set(H.sum(axis=0).tolist()) == {3}
    assert set(H.sum(axis=1).tolist()) == {6}
    with pytest.raises(ValueError):
        gallager_ldpc_matrix(61)


def test_bit_flipping_decodes_sparse_errors_on_an_ldpc_code():
    H = gallager_ldpc_matrix(240, seed=1)
    rng = np.random.default_rng(5)
    successes = 0
    for _ in range(30):
        received = np.zeros(240, dtype=np.uint8)
        received[rng.choice(240, size=3, replace=False)] = 1
        r = bit_flip_decode(H, received)
        if r.converged:
            assert not ((H.astype(int) @ r.codeword) % 2).any()
            successes += not r.codeword.any()
    assert successes >= 27
    stuck = bit_flip_decode(H, np.eye(1, 240, 0, dtype=np.uint8)[0], max_iter=0)
    assert not stuck.converged and stuck.iterations == 0
