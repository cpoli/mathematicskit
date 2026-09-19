r"""The discrete Fourier transform: naive :math:`O(n^2)` DFT, a
hand-rolled radix-2 FFT, and ``numpy.fft`` -- compared directly.

``numpy.fft.fft``/``scipy.fft.fft`` are the primary API for actually
computing a DFT (:func:`fft_numpy`); the naive DFT and radix-2 FFT are
kept hand-rolled specifically as a *pedagogical comparison* -- showing
the :math:`O(n^2)` vs. :math:`O(n\log n)` speedup and cross-checking
correctness -- which is this domain's one deliberate exception to
"call the library directly". See Cooley & Tukey (1965), *An algorithm
for the machine calculation of complex Fourier series*, Math. Comp. 19,
and Cormen et al., *Introduction to Algorithms*, 3rd ed., Ch. 30.
"""

from __future__ import annotations

import time

import numpy as np

from mathematicskit.special_functions.core.base import FFTComparisonResult

__all__ = ["dft_naive", "fft_radix2", "fft_numpy", "compare_fft_methods"]


def dft_naive(x: np.ndarray) -> np.ndarray:
    r"""The discrete Fourier transform, by direct evaluation of its defining sum.

    :math:`X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}`, computed as a
    dense matrix-vector product -- :math:`O(N^2)` time. Kept hand-rolled
    purely to demonstrate the speedup :func:`fft_radix2` achieves over
    it. See Cormen et al., *Introduction to Algorithms*, 3rd ed.,
    Ch. 30.1.

    Parameters
    ----------
    x : ndarray
        Complex (or real) input signal, any length.

    Returns
    -------
    ndarray, complex

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> np.allclose(dft_naive(x), np.fft.fft(x))
    True
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    w = np.exp(-2j * np.pi * k * j / n)
    return w @ x


def fft_radix2(x: np.ndarray) -> np.ndarray:
    r"""The Cooley-Tukey radix-2 fast Fourier transform.

    Recursively splits the DFT of length :math:`N` (a power of 2) into
    two DFTs of length :math:`N/2` (even- and odd-indexed samples),
    combined via the "butterfly" identity :math:`X_k = E_k + e^{-2\pi
    ik/N}O_k`, :math:`X_{k+N/2} = E_k - e^{-2\pi ik/N}O_k` -- giving
    :math:`O(N\log N)` time overall. Requires ``len(x)`` to be a power
    of 2. See Cooley & Tukey (1965), Math. Comp. 19, and Cormen et al.,
    *Introduction to Algorithms*, 3rd ed., Ch. 30.2.

    Parameters
    ----------
    x : ndarray
        Complex (or real) input signal; ``len(x)`` must be a power of 2.

    Returns
    -------
    ndarray, complex

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    >>> np.allclose(fft_radix2(x), np.fft.fft(x))
    True
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    if n & (n - 1) != 0:
        raise ValueError(f"fft_radix2 requires a power-of-2 length, got {n}")
    return _fft_radix2_recursive(x)


def _fft_radix2_recursive(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return x
    even = _fft_radix2_recursive(x[0::2])
    odd = _fft_radix2_recursive(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    half = twiddle * odd
    return np.concatenate([even + half, even - half])


def fft_numpy(x: np.ndarray) -> np.ndarray:
    """The discrete Fourier transform via :func:`numpy.fft.fft`.

    The primary API for actually computing a DFT in this domain: a
    highly optimized mixed-radix FFT (handling any length, not just
    powers of 2). See NumPy's FFT documentation.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    ndarray, complex

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose(fft_numpy(np.array([1.0, 0.0, -1.0, 0.0])), np.fft.fft([1.0, 0.0, -1.0, 0.0]))
    True
    """
    return np.fft.fft(x)


def compare_fft_methods(n: int, seed: int = 0) -> FFTComparisonResult:
    r"""Time and cross-check the naive DFT, radix-2 FFT, and ``numpy.fft`` on the same random signal.

    Demonstrates the :math:`O(n^2)` vs. :math:`O(n\log n)` gap directly:
    the naive DFT's time grows quadratically with `n` while the radix-2
    FFT's (and ``numpy.fft``'s) grows only log-linearly, becoming
    dramatically faster for even moderately large `n`.

    Parameters
    ----------
    n : int
        Signal length; must be a power of 2 (for :func:`fft_radix2`).
    seed : int

    Returns
    -------
    FFTComparisonResult

    Examples
    --------
    >>> result = compare_fft_methods(256, seed=0)
    >>> result.max_error_naive_vs_numpy < 1e-8
    True
    >>> result.max_error_radix2_vs_numpy < 1e-8
    True
    """
    if n & (n - 1) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)

    t0 = time.perf_counter()
    naive_result = dft_naive(x)
    naive_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    radix2_result = fft_radix2(x)
    radix2_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    numpy_result = fft_numpy(x)
    numpy_time = time.perf_counter() - t0

    return FFTComparisonResult(
        naive_result=naive_result,
        radix2_result=radix2_result,
        numpy_result=numpy_result,
        naive_time=naive_time,
        radix2_time=radix2_time,
        numpy_time=numpy_time,
        max_error_naive_vs_numpy=float(np.max(np.abs(naive_result - numpy_result))),
        max_error_radix2_vs_numpy=float(np.max(np.abs(radix2_result - numpy_result))),
    )
