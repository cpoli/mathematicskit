r"""Laurent series coefficients from samples on a circle, via :func:`numpy.fft.fft`.

On an annulus :math:`\rho_1 < |z - z_0| < \rho_2` where :math:`f` is
holomorphic, :math:`f(z) = \sum_{k=-\infty}^{\infty} a_k (z - z_0)^k` with

.. math::

   a_k = \frac{1}{2\pi i}\oint_{|z - z_0| = r} \frac{f(z)}{(z - z_0)^{k+1}}\,dz
       = \frac{1}{2\pi r^k}\int_0^{2\pi} f(z_0 + re^{it})\,e^{-ikt}\,dt.

The trapezoidal rule on :math:`N` equally spaced points turns the
right-hand side into a discrete Fourier transform, so one FFT gives
every coefficient at once, with only aliasing error
:math:`a_{k \pm N} r^{\pm N}`. See P. A. Laurent's 1843 memoir, reported
in A.-L. Cauchy, "Rapport sur un mémoire de M. Laurent," *Comptes
rendus* 17 (1843), 938-942; P. Henrici, *Applied and Computational
Complex Analysis*, vol. 3 (Wiley, 1986), Sec. 13.2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mathematicskit.complex_analysis.core.base import LaurentSeriesResult

__all__ = ["laurent_coefficients"]


def laurent_coefficients(f: Callable, z0: complex, radius: float, n_max: int, n_points: int = 256) -> LaurentSeriesResult:
    r"""Laurent coefficients :math:`a_{-n}, \ldots, a_n` of ``f`` about ``z0``, sampled on :math:`|z - z_0| = r`.

    The circle must lie inside the annulus of convergence being studied:
    the same ``f`` has different Laurent series on different annuli.

    Parameters
    ----------
    f : callable
        Vectorized ``f(z) -> complex``.
    z0 : complex
    radius : float
    n_max : int
        Largest :math:`|k|` returned; must be less than ``n_points // 2``.
    n_points : int

    Returns
    -------
    LaurentSeriesResult

    Examples
    --------
    >>> series = laurent_coefficients(lambda z: np.exp(z) / z**2, 0.0, 1.0, 3)
    >>> [round(series.coefficient(k).real, 10) for k in (-2, -1, 0, 1)]  # 1/z^2 + 1/z + 1/2 + z/6
    [1.0, 1.0, 0.5, 0.1666666667]
    """
    if not 0 < n_max < n_points // 2:
        raise ValueError(f"need 0 < n_max < n_points // 2, got n_max={n_max}, n_points={n_points}")
    t = 2.0 * np.pi * np.arange(n_points) / n_points
    c = np.fft.fft(np.asarray(f(z0 + radius * np.exp(1j * t)), dtype=complex)) / n_points
    orders = np.arange(-n_max, n_max + 1)
    return LaurentSeriesResult(center=complex(z0), radius=float(radius), orders=orders, coefficients=c[orders % n_points] / radius ** orders.astype(float))
