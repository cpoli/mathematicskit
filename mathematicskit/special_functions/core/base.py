"""Result containers for mathematicskit.special_functions.

:class:`FFTComparisonResult` is the shared result type for this domain's
one deliberately hand-rolled comparison (naive DFT vs. radix-2 FFT vs.
``numpy.fft``); the gamma/beta, Bessel, and orthogonal-polynomial
``systems/`` modules are thin, well-documented wrappers around
``scipy.special``/``numpy.polynomial`` and don't need a dataclass beyond
the plain arrays/floats those libraries already return. The exceptions
are functions that ``scipy.special`` returns as a *tuple* of related
arrays (Jacobi elliptic functions, Airy functions, Fresnel integrals),
which get a small named container instead of a bare tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FFTComparisonResult", "JacobiEllipticResult", "AiryResult", "FresnelResult"]


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


@dataclass
class JacobiEllipticResult:
    r"""Jacobi elliptic functions :math:`\operatorname{sn}`, :math:`\operatorname{cn}`, :math:`\operatorname{dn}` at ``(u, m)``."""

    sn: np.ndarray
    """ndarray: :math:`\\operatorname{sn}(u\\mid m) = \\sin\\varphi`."""

    cn: np.ndarray
    """ndarray: :math:`\\operatorname{cn}(u\\mid m) = \\cos\\varphi`."""

    dn: np.ndarray
    """ndarray: :math:`\\operatorname{dn}(u\\mid m) = \\sqrt{1 - m\\sin^2\\varphi}`."""

    amplitude: np.ndarray
    """ndarray: The Jacobi amplitude :math:`\\varphi = \\operatorname{am}(u\\mid m)`."""


@dataclass
class AiryResult:
    """The Airy functions ``Ai``, ``Bi`` and their derivatives at the sample points ``x``."""

    x: np.ndarray
    ai: np.ndarray
    ai_prime: np.ndarray
    bi: np.ndarray
    bi_prime: np.ndarray


@dataclass
class FresnelResult:
    r"""The Fresnel integrals :math:`S(t)` and :math:`C(t)` at the sample points ``t``."""

    t: np.ndarray
    s: np.ndarray
    """ndarray: :math:`S(t) = \\int_0^t \\sin(\\pi\\tau^2/2)\\,d\\tau`."""

    c: np.ndarray
    """ndarray: :math:`C(t) = \\int_0^t \\cos(\\pi\\tau^2/2)\\,d\\tau`."""
