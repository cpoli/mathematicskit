r"""Convolution and correlation: direct sums, the convolution theorem, and FFT convolution.

The linear convolution :math:`(x * h)[n] = \sum_k x[k]\,h[n-k]` costs
:math:`O(NM)` evaluated directly (:func:`numpy.convolve`). The
convolution theorem, :math:`\mathcal{F}\{x * h\} = \mathcal{F}\{x\}\cdot
\mathcal{F}\{h\}`, turns it into a pointwise product of spectra. Zero-padded
to length :math:`N + M - 1` and computed with the FFT, it costs
:math:`O((N+M)\log(N+M))` (:func:`scipy.signal.fftconvolve`). Without the
padding, the product of length-:math:`N` DFTs gives the *circular*
convolution instead. See T. G. Stockham Jr., "High-speed convolution and
correlation," AFIPS Spring Joint Computer Conference 28 (1966), 229-233,
and Oppenheim & Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
Sec. 8.6-8.7.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import signal

from mathematicskit.special_functions.core.base import ConvolutionComparisonResult

__all__ = ["convolve_direct", "convolve_fft", "circular_convolve", "cross_correlate", "compare_convolution_methods"]


def convolve_direct(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    r"""Linear convolution by the direct :math:`O(NM)` sum, via :func:`numpy.convolve`.

    Parameters
    ----------
    x, h : ndarray
        1-D input signal and kernel.
    mode : {"full", "same", "valid"}
        Output length: ``N + M - 1``, ``max(N, M)``, or ``max(N, M) - min(N, M) + 1``.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> convolve_direct([1.0, 2.0, 3.0], [0.0, 1.0, 0.5])
    array([0. , 1. , 2.5, 4. , 1.5])
    """
    return np.convolve(x, h, mode=mode)  # type: ignore[call-overload]  # numpy stubs want Literal mode


def convolve_fft(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    r"""Linear convolution via the convolution theorem, using :func:`scipy.signal.fftconvolve`.

    Both inputs are zero-padded to length :math:`N + M - 1` so the
    circular convolution the DFT computes equals the linear one.

    Parameters
    ----------
    x, h : ndarray
        1-D input signal and kernel.
    mode : {"full", "same", "valid"}

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose(convolve_fft([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]), [0.0, 1.0, 2.5, 4.0, 1.5])
    True
    """
    return signal.fftconvolve(x, h, mode=mode)


def circular_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    r"""Circular (periodic) convolution of two equal-length sequences.

    :math:`(x \circledast h)[n] = \sum_{k=0}^{N-1} x[k]\,h[(n-k) \bmod N]
    = \mathcal{F}^{-1}\{X_k H_k\}`, the convolution theorem for the DFT
    applied directly with :func:`numpy.fft.fft`. Real inputs give a real
    output.

    Parameters
    ----------
    x, h : ndarray
        1-D sequences of the same length.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose(circular_convolve([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 0.0]), [4.0, 1.0, 2.0, 3.0])
    True
    """
    x = np.asarray(x)
    h = np.asarray(h)
    if x.shape != h.shape or x.ndim != 1:
        raise ValueError(f"circular_convolve requires two 1-D sequences of equal length, got shapes {x.shape} and {h.shape}")
    result = np.fft.ifft(np.fft.fft(x) * np.fft.fft(h))
    if np.isrealobj(x) and np.isrealobj(h):
        return result.real
    return result


def cross_correlate(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
    r"""Cross-correlation :math:`(x \star y)[k] = \sum_n x[n+k]\,\overline{y[n]}`, via :func:`scipy.signal.correlate`.

    Correlation is convolution with the conjugated, time-reversed second
    signal, so its peak locates the lag at which ``y`` best matches ``x``.
    With ``mode="full"``, output index ``i`` corresponds to lag
    ``i - (len(y) - 1)`` (see :func:`scipy.signal.correlation_lags`).

    Parameters
    ----------
    x, y : ndarray
    mode : {"full", "same", "valid"}

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np
    >>> c = cross_correlate([0.0, 0.0, 1.0, 2.0, 0.0], [1.0, 2.0])
    >>> int(np.argmax(c)) - 1  # lag of the best match
    2
    """
    return signal.correlate(x, y, mode=mode, method="auto")


def compare_convolution_methods(n: int, m: int, seed: int = 0) -> ConvolutionComparisonResult:
    r"""Time and cross-check direct and FFT convolution of random signals of lengths ``n`` and ``m``.

    Direct convolution scales as :math:`O(nm)`; FFT convolution as
    :math:`O((n+m)\log(n+m))`, so the FFT wins once the kernel is more
    than a few dozen samples long.

    Parameters
    ----------
    n, m : int
        Signal and kernel lengths.
    seed : int

    Returns
    -------
    ConvolutionComparisonResult

    Examples
    --------
    >>> result = compare_convolution_methods(1000, 200, seed=0)
    >>> result.max_error < 1e-9
    True
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    h = rng.normal(size=m)

    t0 = time.perf_counter()
    direct = np.convolve(x, h)
    direct_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fft = signal.fftconvolve(x, h)
    fft_time = time.perf_counter() - t0

    return ConvolutionComparisonResult(
        direct_result=direct,
        fft_result=fft,
        direct_time=direct_time,
        fft_time=fft_time,
        max_error=float(np.max(np.abs(direct - fft))),
    )
