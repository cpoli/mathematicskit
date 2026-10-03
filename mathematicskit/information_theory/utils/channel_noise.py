"""Simulated channel noise and Hamming distance, shared by the channel-coding
modules and their examples."""

from __future__ import annotations

import numpy as np

__all__ = ["bsc_transmit", "hamming_distance"]


def bsc_transmit(bits, p: float, seed=None) -> np.ndarray:
    """Send `bits` through a binary symmetric channel that flips each bit with probability `p`.

    Parameters
    ----------
    bits : array_like of {0, 1}
    p : float
        Crossover probability.
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    ndarray of uint8

    Examples
    --------
    >>> bsc_transmit([0, 1, 1, 0], p=0.0).tolist()
    [0, 1, 1, 0]
    >>> bsc_transmit([0, 1, 1, 0], p=1.0).tolist()
    [1, 0, 0, 1]
    """
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    bits = np.asarray(bits, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return bits ^ (rng.random(bits.shape) < p).astype(np.uint8)


def hamming_distance(a, b) -> int:
    """Number of positions where `a` and `b` differ.

    Parameters
    ----------
    a, b : array_like
        Equal-length sequences.

    Returns
    -------
    int

    Examples
    --------
    >>> hamming_distance([1, 0, 1, 1], [1, 1, 1, 0])
    2
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("sequences must have the same length")
    return int(np.count_nonzero(a != b))
