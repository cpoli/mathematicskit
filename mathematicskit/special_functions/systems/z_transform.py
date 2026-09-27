r"""The Z-transform: evaluation, poles and zeros, and inversion by partial fractions.

The (unilateral) Z-transform of a sequence is
:math:`X(z) = \sum_{n\ge 0} x[n]\,z^{-n}`. A linear constant-coefficient
difference equation :math:`\sum_k a_k y[n-k] = \sum_k b_k x[n-k]` has the
rational transfer function :math:`H(z) = B(z)/A(z)` in powers of
:math:`z^{-1}`. Its poles govern the impulse response: the system is
stable exactly when every pole lies inside the unit circle, and on the
unit circle :math:`z = e^{i\omega}` the transform reduces to the
discrete-time Fourier transform (see
:func:`~mathematicskit.special_functions.systems.filters.frequency_response`).
See J. R. Ragazzini and L. A. Zadeh, "The analysis of sampled-data
systems," Trans. AIEE 71 (1952), 225-234, and Oppenheim & Schafer,
*Discrete-Time Signal Processing*, 3rd ed., Ch. 3.

Poles/zeros and the partial-fraction expansion come from
:func:`scipy.signal.tf2zpk` and :func:`scipy.signal.residuez`; evaluating
the defining sum and inverting each partial fraction term are hand-rolled
since scipy has no routine for either.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import comb

from mathematicskit.special_functions.core.base import PoleZeroResult

__all__ = ["z_transform", "transfer_function", "poles_zeros", "inverse_z_transform"]


def z_transform(x: np.ndarray, z) -> np.ndarray:
    r"""The Z-transform :math:`X(z) = \sum_{n=0}^{N-1} x[n]\,z^{-n}` of a finite sequence.

    Evaluated as a polynomial in :math:`z^{-1}` by Horner's rule
    (:func:`numpy.polyval`).

    Parameters
    ----------
    x : ndarray
        Sequence :math:`x[0], \ldots, x[N-1]`.
    z : complex or ndarray
        Evaluation point(s); must be nonzero.

    Returns
    -------
    complex or ndarray

    Examples
    --------
    >>> complex(z_transform([1.0, 2.0, 3.0], 2.0))  # 1 + 2/2 + 3/4
    (2.75+0j)
    """
    x = np.asarray(x)
    return np.polyval(x[::-1], 1.0 / np.asarray(z, dtype=np.complex128))


def transfer_function(b: np.ndarray, a: np.ndarray, z) -> np.ndarray:
    r"""Evaluate the rational transfer function :math:`H(z) = B(z)/A(z)` in powers of :math:`z^{-1}`.

    Parameters
    ----------
    b, a : ndarray
        Numerator and denominator coefficients, as in :func:`scipy.signal.lfilter`.
    z : complex or ndarray

    Returns
    -------
    complex or ndarray

    Examples
    --------
    >>> complex(transfer_function([1.0], [1.0, -0.5], 2.0))  # 1 / (1 - 0.5/2)
    (1.3333333333333333+0j)
    """
    return z_transform(b, z) / z_transform(a, z)


def poles_zeros(b: np.ndarray, a: np.ndarray) -> PoleZeroResult:
    r"""Poles, zeros, and gain of :math:`H(z) = B(z)/A(z)`, and whether the causal system is stable.

    ``b`` and ``a`` are zero-padded to equal length before calling
    :func:`scipy.signal.tf2zpk`, so that zeros or poles at :math:`z = 0`
    implied by the :math:`z^{-1}` convention are kept.

    Parameters
    ----------
    b, a : ndarray

    Returns
    -------
    PoleZeroResult

    Examples
    --------
    >>> result = poles_zeros([1.0], [1.0, -0.9])
    >>> result.poles, result.zeros, result.is_stable
    (array([0.9]), array([0.]), True)
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = max(b.size, a.size)
    b = np.pad(b, (0, n - b.size))
    a = np.pad(a, (0, n - a.size))
    zeros, poles, gain = signal.tf2zpk(b, a)
    return PoleZeroResult(zeros=zeros, poles=poles, gain=float(gain), is_stable=bool(np.all(np.abs(poles) < 1.0)))


def inverse_z_transform(b: np.ndarray, a: np.ndarray, n: int) -> np.ndarray:
    r"""The first ``n`` samples of the causal sequence whose Z-transform is :math:`B(z)/A(z)`.

    Expands :math:`H(z)` in partial fractions with
    :func:`scipy.signal.residuez`,

    .. math::

       H(z) = \sum_i \frac{r_i}{(1 - p_i z^{-1})^{m_i}} + \sum_j k_j z^{-j},

    then inverts term by term using the table pair
    :math:`1/(1 - p z^{-1})^m \leftrightarrow \binom{n+m-1}{m-1} p^n`
    (Oppenheim & Schafer, Sec. 3.4). Repeated poles (within
    ``residuez``'s default tolerance, ``1e-3``) take increasing powers
    :math:`m_i = 1, 2, \ldots` in order.

    Parameters
    ----------
    b, a : ndarray
    n : int
        Number of samples :math:`h[0], \ldots, h[n-1]`.

    Returns
    -------
    ndarray
        Real when ``b`` and ``a`` are real.

    Examples
    --------
    >>> import numpy as np
    >>> np.round(inverse_z_transform([1.0], [1.0, -0.5], 4), 12)  # 0.5**n
    array([1.   , 0.5  , 0.25 , 0.125])
    """
    residues, poles, direct = signal.residuez(b, a)
    k = np.arange(n)
    h = np.zeros(n, dtype=np.complex128)
    multiplicity = 0
    for i, (r, p) in enumerate(zip(residues, poles, strict=False)):
        multiplicity = multiplicity + 1 if i > 0 and abs(p - poles[i - 1]) < 1e-3 else 1
        h += r * comb(k + multiplicity - 1, multiplicity - 1) * p**k
    m = min(len(direct), n)
    h[:m] += direct[:m]
    if np.isrealobj(b) and np.isrealobj(a):
        return h.real
    return h
