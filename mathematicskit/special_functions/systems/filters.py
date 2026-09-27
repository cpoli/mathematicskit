r"""Digital filter design and application: Butterworth, Chebyshev, and window-method FIR filters.

Every design routine wraps :mod:`scipy.signal` directly and returns a
:class:`~mathematicskit.special_functions.core.base.FilterCoefficients`
holding the transfer function :math:`H(z) = B(z)/A(z)` in powers of
:math:`z^{-1}`.

- Butterworth (IIR): maximally flat magnitude,
  :math:`|H(i\Omega)|^2 = 1/(1 + (\Omega/\Omega_c)^{2N})` in the analog
  prototype, mapped to discrete time by the bilinear transform
  (:func:`scipy.signal.butter`). See S. Butterworth, "On the theory of
  filter amplifiers," Wireless Engineer 7 (1930), 536-541.
- Chebyshev type I (IIR): equiripple passband, trading ripple for a
  steeper transition (:func:`scipy.signal.cheby1`).
- Window-method FIR: the ideal (sinc) impulse response truncated by a
  window (:func:`scipy.signal.firwin`); the Kaiser window's :math:`\beta`
  trades main-lobe width against side-lobe level. See J. F. Kaiser,
  "Nonrecursive digital filter design using the I0-sinh window
  function," Proc. IEEE Int. Symp. Circuits and Systems (1974), 20-23.

Cutoffs are normalized to the Nyquist frequency (``0 < cutoff < 1``)
unless a sampling rate ``fs`` is given, in which case they are in the
same units as ``fs``, exactly as in :mod:`scipy.signal`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal

from mathematicskit.special_functions.core.base import FilterCoefficients, FrequencyResponseResult

__all__ = ["butterworth_filter", "chebyshev1_filter", "fir_window_filter", "apply_filter", "frequency_response"]


def butterworth_filter(order: int, cutoff, btype: str = "lowpass", fs: Optional[float] = None) -> FilterCoefficients:
    r"""Design a digital Butterworth filter via :func:`scipy.signal.butter`.

    Parameters
    ----------
    order : int
        Filter order :math:`N`; the stopband rolls off at :math:`20N` dB/decade.
    cutoff : float or (float, float)
        -3 dB frequency (a pair for ``"bandpass"``/``"bandstop"``).
    btype : {"lowpass", "highpass", "bandpass", "bandstop"}
    fs : float, optional
        Sampling rate; when omitted, ``cutoff`` is normalized to Nyquist.

    Returns
    -------
    FilterCoefficients

    Examples
    --------
    >>> import numpy as np
    >>> filt = butterworth_filter(4, 0.25)
    >>> response = frequency_response(filt, worN=[0.25 * np.pi])
    >>> round(float(response.magnitude[0] ** 2), 6)  # half power at the cutoff
    0.5
    """
    b, a = signal.butter(order, cutoff, btype=btype, fs=fs)
    return FilterCoefficients(b=np.asarray(b), a=np.asarray(a))


def chebyshev1_filter(order: int, ripple_db: float, cutoff, btype: str = "lowpass", fs: Optional[float] = None) -> FilterCoefficients:
    r"""Design a digital Chebyshev type I filter via :func:`scipy.signal.cheby1`.

    The passband gain ripples between :math:`1` and
    :math:`10^{-r/20}` (``r = ripple_db``); ``cutoff`` is the frequency
    where the gain last reaches :math:`10^{-r/20}`.

    Parameters
    ----------
    order : int
    ripple_db : float
        Peak-to-peak passband ripple, in dB.
    cutoff : float or (float, float)
    btype : {"lowpass", "highpass", "bandpass", "bandstop"}
    fs : float, optional

    Returns
    -------
    FilterCoefficients

    Examples
    --------
    >>> import numpy as np
    >>> filt = chebyshev1_filter(5, 1.0, 0.3)
    >>> gain_db = 20 * np.log10(frequency_response(filt, worN=[0.3 * np.pi]).magnitude[0])
    >>> round(float(gain_db), 6)
    -1.0
    """
    b, a = signal.cheby1(order, ripple_db, cutoff, btype=btype, fs=fs)
    return FilterCoefficients(b=np.asarray(b), a=np.asarray(a))


def fir_window_filter(
    numtaps: int,
    cutoff,
    window: str | tuple = "hamming",
    pass_zero: bool | str = True,
    fs: Optional[float] = None,
) -> FilterCoefficients:
    r"""Design a linear-phase FIR filter by the window method, via :func:`scipy.signal.firwin`.

    The ideal brick-wall impulse response, a sampled sinc, is truncated to
    ``numtaps`` samples and multiplied by ``window``. The result is
    symmetric, so its phase is exactly linear (a pure delay of
    ``(numtaps - 1) / 2`` samples).

    Parameters
    ----------
    numtaps : int
        Filter length (odd for a highpass).
    cutoff : float or sequence of float
    window : str or tuple
        Any :func:`scipy.signal.get_window` spec, e.g. ``"hamming"`` or ``("kaiser", 8.0)``.
    pass_zero : bool or str
        ``True`` for lowpass-style (passes DC), ``False`` for highpass-style,
        or a band type string as in :func:`scipy.signal.firwin`.
    fs : float, optional

    Returns
    -------
    FilterCoefficients
        With ``a == [1.0]``.

    Examples
    --------
    >>> import numpy as np
    >>> filt = fir_window_filter(31, 0.4)
    >>> bool(np.allclose(filt.b, filt.b[::-1]))  # symmetric => linear phase
    True
    """
    b = signal.firwin(numtaps, cutoff, window=window, pass_zero=pass_zero, fs=fs)
    return FilterCoefficients(b=np.asarray(b), a=np.array([1.0]))


def apply_filter(filt: FilterCoefficients, x: np.ndarray, zero_phase: bool = False) -> np.ndarray:
    r"""Run a signal through a filter by its difference equation.

    Solves :math:`\sum_k a_k\,y[n-k] = \sum_k b_k\,x[n-k]` with zero
    initial conditions (:func:`scipy.signal.lfilter`), or filters forward
    and then backward for zero phase distortion
    (:func:`scipy.signal.filtfilt`, squaring the magnitude response).

    Parameters
    ----------
    filt : FilterCoefficients
    x : ndarray
    zero_phase : bool

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np
    >>> moving_average = FilterCoefficients(b=np.ones(3) / 3, a=np.array([1.0]))
    >>> apply_filter(moving_average, np.array([3.0, 3.0, 3.0, 3.0]))
    array([1., 2., 3., 3.])
    """
    if zero_phase:
        return signal.filtfilt(filt.b, filt.a, x)
    return signal.lfilter(filt.b, filt.a, x)


def frequency_response(filt: FilterCoefficients, worN=512, fs: Optional[float] = None) -> FrequencyResponseResult:
    r"""Evaluate :math:`H(e^{i\omega})` on the unit circle via :func:`scipy.signal.freqz`.

    Parameters
    ----------
    filt : FilterCoefficients
    worN : int or array_like
        Number of equally spaced frequencies on :math:`[0, \pi)`, or the
        frequencies themselves (rad/sample, or units of ``fs``).
    fs : float, optional

    Returns
    -------
    FrequencyResponseResult

    Examples
    --------
    >>> import numpy as np
    >>> response = frequency_response(butterworth_filter(2, 0.5), worN=[0.0])
    >>> round(float(response.magnitude[0]), 12)  # unit DC gain
    1.0
    """
    if fs is None:
        w, h = signal.freqz(filt.b, filt.a, worN=worN)
    else:
        w, h = signal.freqz(filt.b, filt.a, worN=worN, fs=fs)
    return FrequencyResponseResult(frequencies=w, response=h, magnitude=np.abs(h), phase=np.unwrap(np.angle(h)))
