"""Result containers for mathkit.special_functions.

:class:`FFTComparisonResult` is the shared result type for this domain's
one deliberately hand-rolled comparison (naive DFT vs. radix-2 FFT vs.
``numpy.fft``); the gamma/beta, Bessel, and orthogonal-polynomial
``systems/`` modules are thin, well-documented wrappers around
``scipy.special``/``numpy.polynomial`` and don't need a dataclass beyond
the plain arrays/floats those libraries already return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FFTComparisonResult"]


@dataclass
class FFTComparisonResult:
    """Container comparing naive DFT, hand-rolled radix-2 FFT, and ``numpy.fft``."""

    naive_result: np.ndarray
    radix2_result: np.ndarray
    numpy_result: np.ndarray

    naive_time: float
    """float: Wall-clock seconds for the naive :math:`O(n^2)` DFT."""

    radix2_time: float
    """float: Wall-clock seconds for the hand-rolled radix-2 FFT."""

    numpy_time: float
    """float: Wall-clock seconds for ``numpy.fft.fft``."""

    max_error_naive_vs_numpy: float
    max_error_radix2_vs_numpy: float
