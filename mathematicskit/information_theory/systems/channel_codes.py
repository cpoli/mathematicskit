r"""Binary error-correcting codes: Hamming codes with syndrome decoding,
convolutional codes with Viterbi decoding, and Gallager's low-density
parity-check codes with bit-flipping decoding.

All hand-rolled over :math:`\mathrm{GF}(2)` with numpy integer arrays;
numpy/scipy have no channel codes. Reed-Solomon codes over larger
fields live in :mod:`mathematicskit.abstract_algebra`. See S. Lin and
D. J. Costello, *Error Control Coding*, 2nd ed. (Pearson, 2004).
"""

from __future__ import annotations

import numpy as np

from mathematicskit.information_theory.core.base import BitFlipResult

__all__ = [
    "hamming_parity_check_matrix",
    "hamming_encode",
    "hamming_decode",
    "convolutional_encode",
    "viterbi_decode",
    "gallager_ldpc_matrix",
    "bit_flip_decode",
]


def _bits(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint8)
    if np.any(x > 1):
        raise ValueError("expected bits (0 or 1)")
    return x


def _hamming_sizes(r: int) -> tuple:
    if r < 2:
        raise ValueError("r must be at least 2")
    n = 2**r - 1
    return n, n - r


def hamming_parity_check_matrix(r: int = 3) -> np.ndarray:
    r"""Parity-check matrix of the :math:`(2^r-1,\ 2^r-r-1)` Hamming code.

    Column :math:`j` (1-based) is the binary expansion of :math:`j`, least
    significant bit in row 0, so the syndrome of a single error spells
    out its position.

    Parameters
    ----------
    r : int
        Number of parity bits, at least 2.

    Returns
    -------
    ndarray of uint8, shape (r, 2**r - 1)

    Examples
    --------
    >>> hamming_parity_check_matrix(3)
    array([[1, 0, 1, 0, 1, 0, 1],
           [0, 1, 1, 0, 0, 1, 1],
           [0, 0, 0, 1, 1, 1, 1]], dtype=uint8)
    """
    n, _ = _hamming_sizes(r)
    positions = np.arange(1, n + 1)
    return ((positions[np.newaxis, :] >> np.arange(r)[:, np.newaxis]) & 1).astype(np.uint8)


def hamming_encode(message, r: int = 3) -> np.ndarray:
    r"""Encode `message` with the Hamming code, block by block.

    Data bits fill the positions that are not powers of two; the parity
    bit at position :math:`2^i` makes every check (row of
    :func:`hamming_parity_check_matrix`) even. R. W. Hamming, "Error
    Detecting and Error Correcting Codes," Bell System Technical Journal
    29(2) (1950), 147-160.

    Parameters
    ----------
    message : array_like of {0, 1}
        Length a multiple of :math:`k = 2^r - r - 1`.
    r : int

    Returns
    -------
    ndarray of uint8
        Length ``len(message) / k * (2**r - 1)``.

    Examples
    --------
    >>> hamming_encode([1, 0, 1, 1]).tolist()
    [0, 1, 1, 0, 0, 1, 1]
    """
    n, k = _hamming_sizes(r)
    message = _bits(message)
    if message.size % k:
        raise ValueError(f"message length must be a multiple of k = {k}")
    blocks = message.reshape(-1, k)
    positions = np.arange(1, n + 1)
    is_parity = (positions & (positions - 1)) == 0
    codewords = np.zeros((blocks.shape[0], n), dtype=np.uint8)
    codewords[:, ~is_parity] = blocks
    H = hamming_parity_check_matrix(r)
    syndromes = (codewords.astype(int) @ H.T.astype(int)) % 2
    codewords[:, np.flatnonzero(is_parity)] = syndromes
    return codewords.ravel()


def hamming_decode(received, r: int = 3) -> np.ndarray:
    """Correct up to one error per block by syndrome decoding and return the message bits.

    Parameters
    ----------
    received : array_like of {0, 1}
        Length a multiple of :math:`2^r - 1`.
    r : int

    Returns
    -------
    ndarray of uint8

    Examples
    --------
    >>> word = hamming_encode([1, 0, 1, 1])
    >>> word[4] ^= 1  # one error
    >>> hamming_decode(word).tolist()
    [1, 0, 1, 1]
    """
    n, _ = _hamming_sizes(r)
    received = _bits(received)
    if received.size % n:
        raise ValueError(f"received length must be a multiple of n = {n}")
    words = received.reshape(-1, n).copy()
    H = hamming_parity_check_matrix(r)
    syndromes = (words.astype(int) @ H.T.astype(int)) % 2
    error_positions = syndromes @ (1 << np.arange(r))
    rows = np.flatnonzero(error_positions)
    words[rows, error_positions[rows] - 1] ^= 1
    positions = np.arange(1, n + 1)
    return words[:, (positions & (positions - 1)) != 0].ravel()


def _conv_outputs(register: int, generators) -> list:
    return [bin(register & g).count("1") & 1 for g in generators]


def convolutional_encode(bits, generators=(0o7, 0o5), constraint_length: int = 3) -> np.ndarray:
    r"""Rate-:math:`1/m` feed-forward convolutional encoder, terminated with :math:`K-1` zeros.

    Each input bit shifts into a :math:`K`-bit register (newest bit most
    significant); output :math:`j` is the parity of the register masked
    by ``generators[j]``. The default is the :math:`(7, 5)_8`, :math:`K=3`
    code with free distance 5. P. Elias, "Coding for Noisy Channels," IRE
    Convention Record 3(4) (1955), 37-46.

    Parameters
    ----------
    bits : array_like of {0, 1}
    generators : sequence of int
        Tap masks, conventionally written in octal.
    constraint_length : int
        :math:`K`.

    Returns
    -------
    ndarray of uint8
        Length ``len(generators) * (len(bits) + K - 1)``.

    Examples
    --------
    >>> convolutional_encode([1, 0, 1, 1]).tolist()
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1]
    """
    K = constraint_length
    state, out = 0, []
    for b in [*_bits(bits).tolist(), *([0] * (K - 1))]:
        register = (b << (K - 1)) | state
        out.extend(_conv_outputs(register, generators))
        state = register >> 1
    return np.array(out, dtype=np.uint8)


def viterbi_decode(received, generators=(0o7, 0o5), constraint_length: int = 3) -> np.ndarray:
    r"""Maximum-likelihood decoding of a terminated convolutional code by the Viterbi algorithm.

    Dynamic programming over the :math:`2^{K-1}`-state trellis: keep, for
    every state, the input path with the smallest Hamming distance to
    the received bits, then trace back from the all-zero final state.
    A. J. Viterbi, "Error Bounds for Convolutional Codes and an
    Asymptotically Optimum Decoding Algorithm," IEEE Transactions on
    Information Theory 13(2) (1967), 260-269.

    Parameters
    ----------
    received : array_like of {0, 1}
        Hard-decision channel output of :func:`convolutional_encode`.
    generators : sequence of int
    constraint_length : int

    Returns
    -------
    ndarray of uint8
        The decoded input bits (termination bits removed).

    Examples
    --------
    >>> word = convolutional_encode([1, 0, 1, 1])
    >>> word[[2, 7]] ^= 1  # two errors, far apart
    >>> viterbi_decode(word).tolist()
    [1, 0, 1, 1]
    """
    K, m = constraint_length, len(generators)
    received = _bits(received)
    if received.size % m:
        raise ValueError(f"received length must be a multiple of {m}")
    steps = received.reshape(-1, m)
    n_states = 2 ** (K - 1)
    transitions = [[_conv_outputs((b << (K - 1)) | s, generators) for b in (0, 1)] for s in range(n_states)]
    metric = np.full(n_states, np.inf)
    metric[0] = 0.0
    history = []
    for y in steps:
        new_metric = np.full(n_states, np.inf)
        back = np.zeros((n_states, 2), dtype=np.int64)
        for s in range(n_states):
            if not np.isfinite(metric[s]):
                continue
            for b in (0, 1):
                nxt = ((b << (K - 1)) | s) >> 1
                cost = metric[s] + np.count_nonzero(np.asarray(transitions[s][b]) != y)
                if cost < new_metric[nxt]:
                    new_metric[nxt] = cost
                    back[nxt] = (s, b)
        metric = new_metric
        history.append(back)
    state, decoded = 0, []
    for back in reversed(history):
        state, b = back[state]
        decoded.append(b)
    decoded.reverse()
    return np.array(decoded[: len(decoded) - (K - 1)], dtype=np.uint8)


def gallager_ldpc_matrix(n: int, column_weight: int = 3, row_weight: int = 6, seed=None) -> np.ndarray:
    r"""Gallager's random regular low-density parity-check matrix.

    The first band of :math:`n/w_r` rows has :math:`w_r` consecutive ones
    each; the other :math:`w_c - 1` bands are random column permutations
    of it. Every column then has weight :math:`w_c` and every row weight
    :math:`w_r`, for a design rate of :math:`1 - w_c/w_r`. R. G. Gallager,
    "Low-Density Parity-Check Codes," IRE Transactions on Information
    Theory 8(1) (1962), 21-28; *Low-Density Parity-Check Codes* (MIT
    Press, 1963).

    Parameters
    ----------
    n : int
        Block length, a multiple of `row_weight`.
    column_weight, row_weight : int
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    ndarray of uint8, shape (n * column_weight / row_weight, n)

    Examples
    --------
    >>> H = gallager_ldpc_matrix(12, column_weight=3, row_weight=4, seed=0)
    >>> H.shape, set(H.sum(axis=0).tolist()), set(H.sum(axis=1).tolist())
    ((9, 12), {3}, {4})
    """
    if n % row_weight:
        raise ValueError("n must be a multiple of row_weight")
    rng = np.random.default_rng(seed)
    band = np.zeros((n // row_weight, n), dtype=np.uint8)
    for i in range(n // row_weight):
        band[i, i * row_weight : (i + 1) * row_weight] = 1
    return np.vstack([band] + [band[:, rng.permutation(n)] for _ in range(column_weight - 1)])


def bit_flip_decode(H, received, max_iter: int = 50) -> BitFlipResult:
    r"""Gallager's bit-flipping decoder for a low-density parity-check code.

    While some parity check fails, flip every bit that takes part in the
    largest number of failed checks (Gallager 1963, Section 4.1, decoding
    algorithm A in its simplest hard-decision form).

    Parameters
    ----------
    H : array_like of {0, 1}, shape (m, n)
        Parity-check matrix.
    received : array_like of {0, 1}, shape (n,)
    max_iter : int

    Returns
    -------
    BitFlipResult

    Examples
    --------
    >>> H = hamming_parity_check_matrix(3)
    >>> r = bit_flip_decode(H, [0, 0, 0, 0, 0, 0, 1])
    >>> r.codeword.tolist(), r.converged
    ([0, 0, 0, 0, 0, 0, 0], True)
    """
    H = _bits(H).astype(np.int64)
    word = _bits(received).copy()
    for iteration in range(max_iter + 1):
        syndrome = (H @ word) % 2
        if not syndrome.any():
            return BitFlipResult(codeword=word, iterations=iteration, converged=True)
        if iteration == max_iter:
            break
        unsatisfied = syndrome @ H
        word[unsatisfied == unsatisfied.max()] ^= 1
    return BitFlipResult(codeword=word, iterations=max_iter, converged=False)
