"""Result containers for mathematicskit.special_functions.

:class:`FFTComparisonResult` is the shared result type for this domain's
one deliberately hand-rolled comparison (naive DFT vs. radix-2 FFT vs.
``numpy.fft``); the gamma/beta, Bessel, and orthogonal-polynomial
``systems/`` modules are thin, well-documented wrappers around
``scipy.special``/``numpy.polynomial`` and don't need a dataclass beyond
the plain arrays/floats those libraries already return. The exceptions
are functions that ``scipy.special`` returns as a *tuple* of related
arrays (Jacobi elliptic functions, Airy functions, Fresnel integrals),
which get a small named container instead of a bare tuple. The signal
and transform modules (convolution, filters, the Z- and Laplace
transforms, wavelets) follow the same rule: filter coefficients, pole-zero
data, frequency responses, wavelet decompositions, and scalograms are
returned as named containers so the visualizers have a stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FFTComparisonResult",
    "JacobiEllipticResult",
    "AiryResult",
    "FresnelResult",
    "ConvolutionComparisonResult",
    "FilterCoefficients",
    "FrequencyResponseResult",
    "PoleZeroResult",
    "WaveletDecompositionResult",
    "ScalogramResult",
]


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


@dataclass
class ConvolutionComparisonResult:
    """Container comparing direct :math:`O(nm)` convolution with FFT-based convolution."""

    direct_result: np.ndarray
    fft_result: np.ndarray

    direct_time: float
    """float: Wall-clock seconds for direct (sliding-sum) convolution."""

    fft_time: float
    """float: Wall-clock seconds for FFT convolution."""

    max_error: float
    """float: ``max |direct - fft|``."""


@dataclass
class FilterCoefficients:
    r"""A digital filter's transfer function :math:`H(z) = B(z)/A(z)`.

    Both polynomials are in powers of :math:`z^{-1}`, following
    :func:`scipy.signal.lfilter`: :math:`B(z) = b_0 + b_1 z^{-1} + \cdots`.
    An FIR filter has ``a == [1.0]``.
    """

    b: np.ndarray
    """ndarray: Numerator (feedforward) coefficients."""

    a: np.ndarray
    """ndarray: Denominator (feedback) coefficients, with ``a[0] == 1``."""


@dataclass
class FrequencyResponseResult:
    r"""A filter's frequency response :math:`H(e^{i\omega})` sampled on :math:`[0, \pi)` (or :math:`[0, f_s/2)`)."""

    frequencies: np.ndarray
    """ndarray: Sample frequencies, in rad/sample, or in the units of ``fs`` when given."""

    response: np.ndarray
    """ndarray, complex: :math:`H(e^{i\\omega})`."""

    magnitude: np.ndarray
    """ndarray: :math:`|H|`."""

    phase: np.ndarray
    """ndarray: Unwrapped phase of :math:`H`, in radians."""


@dataclass
class PoleZeroResult:
    """Poles, zeros, and gain of a rational transfer function in :math:`z`."""

    zeros: np.ndarray
    poles: np.ndarray
    gain: float

    is_stable: bool
    """bool: ``True`` when every pole lies strictly inside the unit circle."""


@dataclass
class WaveletDecompositionResult:
    """A multilevel discrete wavelet decomposition."""

    approximation: np.ndarray
    """ndarray: Coarsest-level approximation (scaling) coefficients."""

    details: list
    """list of ndarray: Detail (wavelet) coefficients, finest level first."""

    wavelet: str
    """str: The wavelet name, e.g. ``"haar"`` or ``"db2"``."""


@dataclass
class ScalogramResult:
    """A continuous wavelet transform sampled on a time-scale grid."""

    times: np.ndarray
    scales: np.ndarray

    frequencies: np.ndarray
    """ndarray: Fourier frequency equivalent to each scale, in cycles per unit time."""

    coefficients: np.ndarray
    """ndarray, complex: Shape ``(len(scales), len(times))``."""
