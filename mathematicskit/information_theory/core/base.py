r"""Result containers shared by mathematicskit.information_theory.

The domain's algorithm families (entropy measures, source codes,
channel capacities, error-correcting codes) don't share a common
interface, so there's no ABC here -- only the dataclasses its
visualizers and examples consume: a prefix code with its symbol
probabilities, a channel-capacity computation, a typical-set count, and
an iterative decoder's outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import entropy as _scipy_entropy

__all__ = ["PrefixCode", "ChannelCapacityResult", "TypicalSetResult", "BitFlipResult"]


@dataclass
class PrefixCode:
    r"""A binary prefix code: no codeword is the beginning of another.

    Parameters
    ----------
    codewords : dict
        ``{symbol: codeword}``, each codeword a string of ``"0"``/``"1"``.
    probabilities : dict
        ``{symbol: probability}`` for the source the code was built for.

    Examples
    --------
    >>> code = PrefixCode({"a": "0", "b": "10", "c": "11"}, {"a": 0.5, "b": 0.25, "c": 0.25})
    >>> code.average_length, code.entropy
    (1.5, 1.5)
    >>> code.encode("abca")
    '010110'
    >>> code.decode("010110")
    ['a', 'b', 'c', 'a']
    """

    codewords: dict = field(default_factory=dict)
    probabilities: dict = field(default_factory=dict)

    def __post_init__(self):
        words = sorted(self.codewords.values())
        for shorter, longer in zip(words, words[1:], strict=False):
            if longer.startswith(shorter):
                raise ValueError(f"{shorter!r} is a prefix of {longer!r}: not a prefix code")

    @property
    def average_length(self) -> float:
        r"""float: Expected codeword length :math:`\bar L = \sum_x p(x)\,\ell(x)` in bits per symbol."""
        return float(sum(p * len(self.codewords[s]) for s, p in self.probabilities.items()))

    @property
    def entropy(self) -> float:
        """float: Entropy of the source in bits, the lower bound on :attr:`average_length`."""
        return float(_scipy_entropy(list(self.probabilities.values()), base=2))

    @property
    def efficiency(self) -> float:
        """float: ``entropy / average_length``, at most 1."""
        return self.entropy / self.average_length

    def encode(self, message) -> str:
        """Concatenate the codewords of the symbols in `message`.

        Parameters
        ----------
        message : iterable
            Symbols of the source alphabet.

        Returns
        -------
        str
        """
        return "".join(self.codewords[s] for s in message)

    def decode(self, bits: str) -> list:
        """Split `bits` into codewords; the prefix property makes this unambiguous.

        Parameters
        ----------
        bits : str

        Returns
        -------
        list
            The decoded symbols.
        """
        lookup = {w: s for s, w in self.codewords.items()}
        symbols, word = [], ""
        for b in bits:
            word += b
            if word in lookup:
                symbols.append(lookup[word])
                word = ""
        if word:
            raise ValueError(f"trailing bits {word!r} are not a complete codeword")
        return symbols


@dataclass
class ChannelCapacityResult:
    """Container for a channel-capacity computation (Blahut-Arimoto)."""

    capacity: float
    """float: Capacity in bits per channel use."""

    input_distribution: np.ndarray
    """ndarray: The capacity-achieving input distribution."""

    iterations: int
    """int: Number of iterations actually performed."""

    converged: bool
    """bool: Whether the upper and lower capacity bounds met within the tolerance."""

    lower_bounds: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray: Lower bound on the capacity at each iteration, in bits."""

    upper_bounds: np.ndarray = field(default_factory=lambda: np.empty(0))
    """ndarray: Upper bound on the capacity at each iteration, in bits."""


@dataclass
class TypicalSetResult:
    r"""Container for the :math:`\varepsilon`-typical set of an i.i.d. source."""

    n: int
    """int: Sequence length."""

    epsilon: float
    """float: Tolerance on :math:`-\\tfrac1n \\log_2 p(x^n)` around the entropy."""

    entropy: float
    """float: Entropy :math:`H` of one source symbol, in bits."""

    log2_size: float
    """float: :math:`\\log_2` of the number of typical sequences."""

    probability: float
    """float: Total probability of the typical set."""


@dataclass
class BitFlipResult:
    """Container for Gallager's bit-flipping decoder."""

    codeword: np.ndarray
    """ndarray: The decoded word (a codeword when ``converged``)."""

    iterations: int
    """int: Number of flipping rounds performed."""

    converged: bool
    """bool: Whether every parity check is satisfied."""
