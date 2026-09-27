r"""Wavelets: Daubechies filters, the discrete wavelet transform, and the Morlet continuous transform.

SciPy removed its wavelet routines (``scipy.signal.daub``, ``cwt``,
``morlet2``) in version 1.15 and numpy never had any, so this module
hand-rolls them rather than adding a PyWavelets dependency.

- :func:`daubechies_filter` builds the orthonormal Daubechies lowpass
  filter with :math:`p` vanishing moments by spectral factorization
  (Daubechies, *Ten Lectures on Wavelets*, SIAM 1992, Sec. 6.1);
  :math:`p = 1` is the Haar filter :math:`(1, 1)/\sqrt2`
  (A. Haar, "Zur Theorie der orthogonalen Funktionensysteme,"
  Math. Ann. 69 (1910), 331-371).
- :func:`discrete_wavelet_transform` is Mallat's pyramid algorithm:
  filter with the lowpass :math:`h` and highpass
  :math:`g[n] = (-1)^n h[L-1-n]`, keep every second sample, and repeat on
  the lowpass half (S. Mallat, "A theory for multiresolution signal
  decomposition," IEEE Trans. PAMI 11 (1989), 674-693). Periodic
  extension makes the transform orthogonal, so it preserves energy and
  :func:`inverse_discrete_wavelet_transform` reconstructs exactly.
- :func:`morlet_cwt` computes the continuous wavelet transform with the
  Morlet wavelet by FFT, following C. Torrence and G. P. Compo, "A
  practical guide to wavelet analysis," Bull. Amer. Meteor. Soc. 79
  (1998), 61-78, after A. Grossmann and J. Morlet, SIAM J. Math. Anal.
  15 (1984), 723-736.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import comb

from mathematicskit.special_functions.core.base import ScalogramResult, WaveletDecompositionResult

__all__ = [
    "daubechies_filter",
    "wavelet_filters",
    "discrete_wavelet_transform",
    "inverse_discrete_wavelet_transform",
    "morlet_cwt",
]


def daubechies_filter(p: int) -> np.ndarray:
    r"""The orthonormal Daubechies lowpass (scaling) filter with ``p`` vanishing moments.

    The filter has length :math:`2p` and frequency response
    :math:`H(\omega) = \sqrt2\bigl(\tfrac{1+e^{-i\omega}}{2}\bigr)^p Q(e^{-i\omega})`,
    where :math:`|Q|^2 = P(\sin^2(\omega/2))`,
    :math:`P(y) = \sum_{k=0}^{p-1}\binom{p-1+k}{k}y^k`. Substituting
    :math:`y = (2 - z - z^{-1})/4` turns :math:`P` into a polynomial whose
    roots come in reciprocal pairs :math:`(r, 1/r)`; keeping the roots
    inside the unit circle gives the minimum-phase :math:`Q`. Root-finding
    becomes ill-conditioned beyond :math:`p \approx 20`.

    Parameters
    ----------
    p : int
        Number of vanishing moments (``"db<p>"``), ``p >= 1``.

    Returns
    -------
    ndarray
        Coefficients :math:`h[0], \ldots, h[2p-1]`, with :math:`\sum h = \sqrt2`
        and :math:`\sum h^2 = 1`.

    Examples
    --------
    >>> import numpy as np
    >>> h = daubechies_filter(2)
    >>> r3 = np.sqrt(3.0)
    >>> bool(np.allclose(h, np.array([1 + r3, 3 + r3, 3 - r3, 1 - r3]) / (4 * np.sqrt(2.0))))
    True
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    q = np.array([1.0])
    if p > 1:
        roots = np.roots(_daubechies_p_polynomial(p))
        q = np.real(np.poly(roots[np.abs(roots) < 1.0]))
    h = np.convolve(comb(p, np.arange(p + 1)), q)
    return h * np.sqrt(2.0) / h.sum()


def _daubechies_p_polynomial(p: int) -> np.ndarray:
    """Coefficients (highest power first) of z^(p-1) * P((2 - z - 1/z)/4), degree 2(p-1)."""
    zy = np.array([-0.25, 0.5, -0.25])  # z * y(z), highest power first
    total = np.zeros(2 * p - 1)
    zy_power = np.array([1.0])
    for k in range(p):
        # z^(p-1) y^k = z^(p-1-k) (z y)^k: shift by p-1-k powers of z
        term = np.concatenate([comb(p - 1 + k, k) * zy_power, np.zeros(p - 1 - k)])
        total[total.size - term.size :] += term
        zy_power = np.convolve(zy_power, zy)
    return total


def wavelet_filters(wavelet: str) -> tuple:
    r"""The orthonormal lowpass/highpass analysis filter pair for a named wavelet.

    Parameters
    ----------
    wavelet : str
        ``"haar"`` (same as ``"db1"``) or ``"db<p>"`` for a Daubechies
        wavelet with ``p`` vanishing moments.

    Returns
    -------
    (ndarray, ndarray)
        The lowpass :math:`h` and the quadrature-mirror highpass
        :math:`g[n] = (-1)^n h[L-1-n]`.

    Examples
    --------
    >>> import numpy as np
    >>> h, g = wavelet_filters("haar")
    >>> bool(np.allclose(h, [2**-0.5, 2**-0.5]) and np.allclose(g, [2**-0.5, -(2**-0.5)]))
    True
    """
    name = wavelet.lower()
    if name == "haar":
        p = 1
    elif name.startswith("db") and name[2:].isdigit():
        p = int(name[2:])
    else:
        raise ValueError(f"unknown wavelet {wavelet!r}; expected 'haar' or 'db<p>'")
    h = daubechies_filter(p)
    g = (-1.0) ** np.arange(h.size) * h[::-1]
    return h, g


def _analysis_indices(n: int, length: int) -> np.ndarray:
    """Periodic index matrix idx[k, j] = (2k + j) mod n."""
    return (2 * np.arange(n // 2)[:, None] + np.arange(length)[None, :]) % n


def discrete_wavelet_transform(x: np.ndarray, wavelet: str = "haar", level: Optional[int] = None) -> WaveletDecompositionResult:
    r"""Multilevel orthogonal discrete wavelet transform with periodic boundary handling.

    Each level computes :math:`a[k] = \sum_j h[j]\,x[(2k+j) \bmod N]` and
    :math:`d[k] = \sum_j g[j]\,x[(2k+j) \bmod N]`, then recurses on
    :math:`a` (Mallat 1989). The transform is orthogonal:
    :math:`\|x\|^2 = \|a\|^2 + \sum \|d_\ell\|^2`.

    Parameters
    ----------
    x : ndarray
        1-D real signal; ``len(x)`` must be divisible by ``2**level``.
    wavelet : str
        See :func:`wavelet_filters`.
    level : int, optional
        Number of levels; defaults to the largest for which every level's
        input is at least as long as the filter.

    Returns
    -------
    WaveletDecompositionResult

    Examples
    --------
    >>> import numpy as np
    >>> result = discrete_wavelet_transform(np.array([4.0, 4.0, 2.0, 0.0]), "haar", level=1)
    >>> np.round(result.approximation * np.sqrt(2), 12), np.round(result.details[0] * np.sqrt(2), 12)
    (array([8., 2.]), array([0., 2.]))
    """
    x = np.asarray(x, dtype=float)
    h, g = wavelet_filters(wavelet)
    n = x.size
    if level is None:
        level = 0
        while n % (2 ** (level + 1)) == 0 and n // 2**level >= max(h.size, 2):
            level += 1
    if level < 1 or n % (2**level) != 0:
        raise ValueError(f"len(x) = {n} must be divisible by 2**level = 2**{level}, with level >= 1")
    details = []
    approx = x
    for _ in range(level):
        idx = _analysis_indices(approx.size, h.size)
        windows = approx[idx]
        details.append(windows @ g)
        approx = windows @ h
    return WaveletDecompositionResult(approximation=approx, details=details, wavelet=wavelet)


def inverse_discrete_wavelet_transform(result: WaveletDecompositionResult) -> np.ndarray:
    r"""Reconstruct a signal from its :func:`discrete_wavelet_transform`.

    Applies the transpose of each (orthogonal) analysis step,
    :math:`x[(2k+j) \bmod N] \mathrel{+}= a[k]\,h[j] + d[k]\,g[j]`, from
    the coarsest level to the finest. Zeroing small detail coefficients
    before reconstructing is wavelet compression/denoising.

    Parameters
    ----------
    result : WaveletDecompositionResult

    Returns
    -------
    ndarray

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(8.0)
    >>> bool(np.allclose(inverse_discrete_wavelet_transform(discrete_wavelet_transform(x, "db2")), x))
    True
    """
    h, g = wavelet_filters(result.wavelet)
    approx = np.asarray(result.approximation, dtype=float)
    for detail in reversed(result.details):
        n = 2 * approx.size
        idx = _analysis_indices(n, h.size)
        x = np.zeros(n)
        np.add.at(x, idx, approx[:, None] * h[None, :] + np.asarray(detail)[:, None] * g[None, :])
        approx = x
    return approx


def morlet_cwt(x: np.ndarray, scales: np.ndarray, dt: float = 1.0, omega0: float = 6.0) -> ScalogramResult:
    r"""Continuous wavelet transform with the Morlet wavelet, computed by FFT.

    The Morlet wavelet :math:`\psi(\eta) = \pi^{-1/4}e^{i\omega_0\eta}e^{-\eta^2/2}`
    has Fourier transform :math:`\hat\psi(\omega) = \pi^{-1/4}e^{-(\omega-\omega_0)^2/2}`
    for :math:`\omega > 0`. At each scale :math:`s`,

    .. math::

       W(s, t_n) = \mathcal{F}^{-1}\Bigl\{\hat x(\omega_k)\,
       \sqrt{2\pi s/\delta t}\;\hat\psi(s\omega_k)\Bigr\}[n]

    (Torrence & Compo 1998, Eqs. 4, 6), a periodic convolution. Scale
    :math:`s` corresponds to Fourier period
    :math:`4\pi s/(\omega_0 + \sqrt{2 + \omega_0^2})`, so a sinusoid of
    frequency :math:`f` peaks near :math:`s \approx 1.03/f` for
    :math:`\omega_0 = 6`.

    Parameters
    ----------
    x : ndarray
        1-D real signal sampled at spacing ``dt``.
    scales : ndarray
        Wavelet scales, in the same time units as ``dt``.
    dt : float
    omega0 : float
        Nondimensional center frequency; ``6`` makes the wavelet
        admissible to double precision.

    Returns
    -------
    ScalogramResult

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(1024) * 0.01
    >>> scales = np.geomspace(0.05, 1.0, 60)
    >>> result = morlet_cwt(np.sin(2 * np.pi * 5.0 * t), scales, dt=0.01)
    >>> peak = np.argmax(np.mean(np.abs(result.coefficients) ** 2, axis=1))
    >>> bool(abs(result.frequencies[peak] - 5.0) < 0.3)
    True
    """
    x = np.asarray(x, dtype=float)
    scales = np.asarray(scales, dtype=float)
    n = x.size
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    x_hat = np.fft.fft(x - x.mean())
    so = scales[:, None] * omega[None, :]
    psi_hat = np.pi**-0.25 * np.exp(-0.5 * (so - omega0) ** 2) * (omega[None, :] > 0)
    coefficients = np.fft.ifft(x_hat[None, :] * np.sqrt(2.0 * np.pi * scales[:, None] / dt) * psi_hat, axis=1)
    fourier_period = 4.0 * np.pi * scales / (omega0 + np.sqrt(2.0 + omega0**2))
    return ScalogramResult(times=np.arange(n) * dt, scales=scales, frequencies=1.0 / fourier_period, coefficients=coefficients)
