r"""Lossless source coding: the typical set behind Shannon's source coding
theorem, the Kraft-McMillan inequality, Shannon-Fano and Huffman codes,
Elias' universal code for the integers, arithmetic coding, and LZ78.

All hand-rolled: numpy/scipy have no entropy coders. Probabilities are
given as a ``{symbol: probability}`` dict or as a sequence, whose
symbols are then the indices ``0, 1, ...``.
"""

from __future__ import annotations

import heapq
import itertools
import math
from fractions import Fraction

import numpy as np
from scipy.special import gammaln

from mathematicskit.information_theory.core.base import PrefixCode, TypicalSetResult
from mathematicskit.information_theory.systems.entropy import entropy

__all__ = [
    "typical_set",
    "kraft_sum",
    "canonical_code",
    "shannon_fano_code",
    "huffman_code",
    "elias_gamma_encode",
    "elias_gamma_decode",
    "arithmetic_intervals",
    "arithmetic_encode",
    "arithmetic_decode",
    "lz78_encode",
    "lz78_decode",
    "lz78_compressed_bits",
]


def _items(probabilities) -> list:
    """``[(symbol, p), ...]`` normalized to sum to 1."""
    pairs = list(probabilities.items()) if isinstance(probabilities, dict) else list(enumerate(probabilities))
    if not pairs:
        raise ValueError("need at least one symbol")
    total = float(sum(p for _, p in pairs))
    if any(p < 0 for _, p in pairs) or total <= 0:
        raise ValueError("probabilities must be non-negative and not all zero")
    return [(s, float(p) / total) for s, p in pairs]


def _compositions(n: int, k: int):
    """Every tuple of `k` non-negative integers summing to `n` (stars and bars)."""
    for bars in itertools.combinations(range(n + k - 1), k - 1):
        edges = (-1, *bars, n + k - 1)
        yield tuple(edges[i + 1] - edges[i] - 1 for i in range(k))


def typical_set(p, n: int, epsilon: float) -> TypicalSetResult:
    r"""Size and probability of the :math:`\varepsilon`-typical set of an i.i.d. source.

    A sequence :math:`x^n` is typical when
    :math:`\bigl|-\tfrac1n\log_2 p(x^n) - H\bigr| \le \varepsilon`. The
    asymptotic equipartition property (Shannon 1948, Theorem 3) says the
    typical set has probability tending to 1 and about :math:`2^{nH}`
    members, so :math:`nH` bits suffice to index the likely sequences --
    the source coding theorem. Computed exactly by summing over sequence
    types (symbol counts) with multinomial coefficients, so the cost
    grows as :math:`\binom{n+k-1}{k-1}`, not :math:`k^n`.

    Parameters
    ----------
    p : array_like
        Symbol probabilities of the source (no zeros).
    n : int
        Sequence length.
    epsilon : float
        Tolerance in bits per symbol.

    Returns
    -------
    TypicalSetResult

    Examples
    --------
    >>> r = typical_set([0.9, 0.1], n=100, epsilon=0.1)
    >>> round(r.probability, 3), round(r.log2_size / 100, 3), round(r.entropy, 3)
    (0.759, 0.529, 0.469)
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("probabilities must be positive")
    p = p / p.sum()
    log2p = np.log2(p)
    H = entropy(p)
    log2_counts, probability = [], 0.0
    for counts in _compositions(n, len(p)):
        c = np.asarray(counts)
        log2_prob_one = float(c @ log2p)
        if abs(-log2_prob_one / n - H) <= epsilon:
            log2_multinomial = (gammaln(n + 1) - gammaln(c + 1).sum()) / np.log(2.0)
            log2_counts.append(log2_multinomial)
            probability += 2.0 ** (log2_multinomial + log2_prob_one)
    log2_size = float(np.logaddexp2.reduce(log2_counts)) if log2_counts else -np.inf
    return TypicalSetResult(n=n, epsilon=epsilon, entropy=H, log2_size=log2_size, probability=float(min(probability, 1.0)))


def kraft_sum(lengths, radix: int = 2) -> float:
    r"""The Kraft sum :math:`\sum_i D^{-\ell_i}` of a list of codeword lengths.

    Kraft (1949) showed a :math:`D`-ary prefix code with lengths
    :math:`\ell_i` exists if and only if the sum is at most 1; McMillan
    (1956) extended the necessity to every uniquely decodable code. See
    L. G. Kraft, *A Device for Quantizing, Grouping, and Coding
    Amplitude-Modulated Pulses*, MS thesis, MIT (1949), and B. McMillan,
    "Two Inequalities Implied by Unique Decipherability," IRE Transactions
    on Information Theory 2(4) (1956), 115-116.

    Parameters
    ----------
    lengths : sequence of int
    radix : int
        Code alphabet size :math:`D`.

    Returns
    -------
    float

    Examples
    --------
    >>> kraft_sum([1, 2, 3, 3])
    1.0
    >>> kraft_sum([1, 1, 2])  # no binary prefix code has these lengths
    1.25
    """
    return float(sum(Fraction(1, radix ** int(l)) for l in lengths))


def canonical_code(lengths) -> list:
    r"""Binary prefix codewords with the given lengths -- the constructive half of Kraft's inequality.

    Assigns codewords in order of increasing length, each the next
    binary number after the previous one, shifted to its length (the
    canonical Huffman code of Schwartz and Kallick, 1964).

    Parameters
    ----------
    lengths : sequence of int
        Codeword lengths with Kraft sum at most 1.

    Returns
    -------
    list of str
        Codewords in the order of `lengths`.

    Examples
    --------
    >>> canonical_code([2, 1, 3, 3])
    ['10', '0', '110', '111']
    """
    if kraft_sum(lengths) > 1:
        raise ValueError("Kraft sum exceeds 1: no prefix code has these lengths")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    words = [""] * len(lengths)
    value, previous = 0, 0
    for i in order:
        value <<= lengths[i] - previous
        words[i] = format(value, f"0{lengths[i]}b") if lengths[i] else ""
        value += 1
        previous = lengths[i]
    return words


def shannon_fano_code(probabilities, method: str = "fano") -> PrefixCode:
    r"""A Shannon-Fano prefix code.

    ``method="fano"`` (R. M. Fano, *The Transmission of Information*,
    MIT RLE Technical Report 65, 1949): sort by probability and split the
    list recursively where the two halves' totals are most nearly equal,
    prefixing ``0`` and ``1``. ``method="shannon"`` (Shannon 1948,
    Section 9): give the :math:`i`-th most likely symbol the first
    :math:`\lceil -\log_2 p_i\rceil` bits of the binary expansion of
    :math:`\sum_{j<i} p_j`. Both are within one bit of the entropy, but
    neither is always optimal (compare :func:`huffman_code`).

    Parameters
    ----------
    probabilities : dict or sequence
    method : {"fano", "shannon"}

    Returns
    -------
    PrefixCode

    Examples
    --------
    >>> shannon_fano_code({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}).codewords
    {'a': '0', 'b': '10', 'c': '110', 'd': '111'}
    >>> shannon_fano_code({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}, method="shannon").codewords
    {'a': '00', 'b': '01', 'c': '101', 'd': '1110'}
    """
    items = sorted(_items(probabilities), key=lambda sp: -sp[1])
    codewords: dict = {}
    if method == "fano":

        def split(group, prefix):
            if len(group) == 1:
                codewords[group[0][0]] = prefix or "0"
                return
            total, running = sum(p for _, p in group), 0.0
            best_k, best_gap = 1, math.inf
            for k in range(1, len(group)):
                running += group[k - 1][1]
                gap = abs(total - 2 * running)
                if gap < best_gap:
                    best_k, best_gap = k, gap
            split(group[:best_k], prefix + "0")
            split(group[best_k:], prefix + "1")

        split(items, "")
    elif method == "shannon":
        cumulative = 0.0
        for s, p in items:
            if p == 0:
                raise ValueError("Shannon's construction needs positive probabilities")
            length = max(math.ceil(-math.log2(p) - 1e-12), 1)
            bits, frac = "", cumulative
            for _ in range(length):
                frac *= 2
                bit = int(frac)
                bits += str(bit)
                frac -= bit
            codewords[s] = bits
            cumulative += p
    else:
        raise ValueError("method must be 'fano' or 'shannon'")
    return PrefixCode(codewords, dict(items))


def huffman_code(probabilities) -> PrefixCode:
    r"""Huffman's optimal binary prefix code.

    Repeatedly merge the two least likely nodes into one whose
    probability is their sum; reading the merges back from the root
    gives each symbol its codeword. No prefix code has a smaller average
    length, which lies in :math:`[H, H+1)`. D. A. Huffman, "A Method for
    the Construction of Minimum-Redundancy Codes," Proceedings of the IRE
    40(9) (1952), 1098-1101.

    Parameters
    ----------
    probabilities : dict or sequence

    Returns
    -------
    PrefixCode

    Examples
    --------
    >>> code = huffman_code({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1})
    >>> sorted((s, len(w)) for s, w in code.codewords.items())
    [('a', 1), ('b', 2), ('c', 3), ('d', 3)]
    >>> round(code.average_length, 2)
    1.9
    """
    items = _items(probabilities)
    if len(items) == 1:
        return PrefixCode({items[0][0]: "0"}, dict(items))
    tie = itertools.count()
    heap = [(p, next(tie), {s: ""}) for s, p in items]
    heapq.heapify(heap)
    while len(heap) > 1:
        p0, _, left = heapq.heappop(heap)
        p1, _, right = heapq.heappop(heap)
        merged = {s: "0" + w for s, w in left.items()} | {s: "1" + w for s, w in right.items()}
        heapq.heappush(heap, (p0 + p1, next(tie), merged))
    codewords = heap[0][2]
    return PrefixCode({s: codewords[s] for s, _ in items}, dict(items))


def elias_gamma_encode(n: int) -> str:
    r"""Elias' gamma code of a positive integer: :math:`\lfloor\log_2 n\rfloor` zeros, then :math:`n` in binary.

    A prefix code for all positive integers needing no prior bound on
    their size, with length :math:`2\lfloor\log_2 n\rfloor + 1`. P. Elias,
    "Universal Codeword Sets and Representations of the Integers," IEEE
    Transactions on Information Theory 21(2) (1975), 194-203.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    str

    Examples
    --------
    >>> [elias_gamma_encode(n) for n in (1, 2, 5, 9)]
    ['1', '010', '00101', '0001001']
    """
    if n < 1:
        raise ValueError("Elias gamma codes positive integers only")
    binary = format(n, "b")
    return "0" * (len(binary) - 1) + binary


def elias_gamma_decode(bits: str) -> list:
    """Decode a concatenation of Elias gamma codewords.

    Parameters
    ----------
    bits : str

    Returns
    -------
    list of int

    Examples
    --------
    >>> elias_gamma_decode("1" "010" "00101")
    [1, 2, 5]
    """
    numbers, i = [], 0
    while i < len(bits):
        zeros = 0
        while i < len(bits) and bits[i] == "0":
            zeros += 1
            i += 1
        if i + zeros + 1 > len(bits):
            raise ValueError("truncated Elias gamma codeword")
        numbers.append(int(bits[i : i + zeros + 1], 2))
        i += zeros + 1
    return numbers


def _cumulative(probabilities):
    """Exact ``{symbol: (low, high)}`` subintervals of [0, 1), in input order."""
    items = _items(probabilities)
    weights = [Fraction(p) for _, p in items]
    total = sum(weights)
    bounds, low = {}, Fraction(0)
    for (s, _), w in zip(items, weights, strict=True):
        bounds[s] = (low, low + w / total)
        low += w / total
    return bounds


def arithmetic_intervals(message, probabilities) -> list:
    r"""The nested intervals of arithmetic coding, one per symbol of `message`.

    Each symbol narrows the current interval :math:`[\ell, h)` to the
    sub-interval its probability occupies, so the final width is
    :math:`p(\text{message})`. Exact rational arithmetic
    (:class:`fractions.Fraction`) avoids the renormalization that
    practical coders need.

    Parameters
    ----------
    message : sequence
    probabilities : dict or sequence

    Returns
    -------
    list of tuple of Fraction
        ``[(0, 1), (low_1, high_1), ...]``, of length ``len(message) + 1``.

    Examples
    --------
    >>> [(float(l), float(h)) for l, h in arithmetic_intervals("ab", {"a": 0.5, "b": 0.5})]
    [(0.0, 1.0), (0.0, 0.5), (0.25, 0.5)]
    """
    bounds = _cumulative(probabilities)
    low, high = Fraction(0), Fraction(1)
    intervals = [(low, high)]
    for s in message:
        width = high - low
        a, b = bounds[s]
        low, high = low + width * a, low + width * b
        intervals.append((low, high))
    return intervals


def arithmetic_encode(message, probabilities) -> str:
    r"""Arithmetic-code `message`: the shortest binary fraction whose dyadic interval fits in the message's interval.

    The code length is at most :math:`\lceil -\log_2 p(\text{message})\rceil + 1`
    bits, within 2 bits of the ideal for the whole message rather than
    within 1 bit per symbol. J. Rissanen, "Generalized Kraft Inequality
    and Arithmetic Coding," IBM Journal of Research and Development 20(3)
    (1976), 198-203; R. C. Pasco, *Source Coding Algorithms for Fast Data
    Compression*, PhD thesis, Stanford (1976).

    Parameters
    ----------
    message : sequence
    probabilities : dict or sequence

    Returns
    -------
    str

    Examples
    --------
    >>> arithmetic_encode("aaab", {"a": 0.75, "b": 0.25})  # p = 0.105, 5 bits
    '01011'
    """
    low, high = arithmetic_intervals(message, probabilities)[-1]
    if low == high:
        raise ValueError("message has probability zero")
    length = max(math.ceil(-math.log2(high - low)), 0)
    while True:
        k = math.ceil(low * 2**length)
        if Fraction(k + 1, 2**length) <= high:
            return format(k, f"0{length}b") if length else ""
        length += 1


def arithmetic_decode(bits: str, probabilities, length: int) -> list:
    """Decode `length` symbols from an arithmetic codeword.

    Parameters
    ----------
    bits : str
        From :func:`arithmetic_encode`.
    probabilities : dict or sequence
        The model used to encode.
    length : int
        Number of symbols in the message.

    Returns
    -------
    list

    Examples
    --------
    >>> "".join(arithmetic_decode("01011", {"a": 0.75, "b": 0.25}, 4))
    'aaab'
    """
    bounds = _cumulative(probabilities)
    value = Fraction(int(bits, 2), 2 ** len(bits)) if bits else Fraction(0)
    low, high = Fraction(0), Fraction(1)
    message = []
    for _ in range(length):
        width = high - low
        for s, (a, b) in bounds.items():
            if low + width * a <= value < low + width * b:
                message.append(s)
                low, high = low + width * a, low + width * b
                break
    return message


def lz78_encode(sequence) -> list:
    r"""Lempel-Ziv (LZ78) parsing into ``(index, symbol)`` pairs.

    Each phrase is the longest previously seen phrase (``index`` into
    the dictionary, 0 for the empty phrase) extended by one new
    ``symbol``. For a stationary ergodic source the compressed length
    per symbol tends to the entropy rate without knowing the source
    statistics. J. Ziv and A. Lempel, "Compression of Individual Sequences
    via Variable-Rate Coding," IEEE Transactions on Information Theory
    24(5) (1978), 530-536.

    Parameters
    ----------
    sequence : sequence

    Returns
    -------
    list of tuple
        ``(index, symbol)``; the last pair's symbol is ``None`` when the
        sequence ends inside an already-known phrase.

    Examples
    --------
    >>> lz78_encode("ABBABBABBBAABABAA")
    [(0, 'A'), (0, 'B'), (2, 'A'), (2, 'B'), (1, 'B'), (4, 'A'), (5, 'A'), (3, 'A')]
    >>> lz78_encode("ABA")  # ends inside the known phrase "A"
    [(0, 'A'), (0, 'B'), (1, None)]
    """
    dictionary: dict = {(): 0}
    pairs: list = []
    phrase: tuple = ()
    for s in sequence:
        if (*phrase, s) in dictionary:
            phrase = (*phrase, s)
        else:
            pairs.append((dictionary[phrase], s))
            dictionary[(*phrase, s)] = len(dictionary)
            phrase = ()
    if phrase:
        pairs.append((dictionary[phrase], None))
    return pairs


def lz78_decode(pairs) -> list:
    """Invert :func:`lz78_encode`.

    Parameters
    ----------
    pairs : list of tuple

    Returns
    -------
    list

    Examples
    --------
    >>> "".join(lz78_decode(lz78_encode("ABBABBABBBAABABAA")))
    'ABBABBABBBAABABAA'
    """
    phrases: list = [()]
    out: list = []
    for index, s in pairs:
        phrase = phrases[index] + ((s,) if s is not None else ())
        out.extend(phrase)
        phrases.append(phrase)
    return out


def lz78_compressed_bits(pairs, alphabet_size: int) -> int:
    r"""Length in bits of an LZ78 parsing: phrase :math:`i` costs :math:`\lceil\log_2 i\rceil + \lceil\log_2 k\rceil` bits.

    Parameters
    ----------
    pairs : list of tuple
        From :func:`lz78_encode`.
    alphabet_size : int
        :math:`k`.

    Returns
    -------
    int

    Examples
    --------
    >>> lz78_compressed_bits(lz78_encode("ABBABBABBBAABABAA"), alphabet_size=2)
    25
    """
    symbol_bits = math.ceil(math.log2(alphabet_size)) if alphabet_size > 1 else 0
    return sum(math.ceil(math.log2(i)) + symbol_bits for i in range(1, len(pairs) + 1))
