r"""Channel capacity and rate-distortion: closed forms for the binary
symmetric, binary erasure and Gaussian channels, the Blahut-Arimoto
algorithm for an arbitrary discrete memoryless channel, a random-coding
experiment illustrating the noisy-channel coding theorem, and the
rate-distortion functions of binary and Gaussian sources.

Blahut-Arimoto and the random-coding simulation are hand-rolled (no
scipy equivalent); the closed forms use :func:`scipy.special.entr` and
:func:`scipy.special.rel_entr`. See Cover and Thomas, *Elements of
Information Theory*, 2nd ed., Chs. 7, 9, 10.
"""

from __future__ import annotations

import numpy as np
from scipy.special import rel_entr

from mathematicskit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from mathematicskit.information_theory.core.base import ChannelCapacityResult
from mathematicskit.information_theory.systems.entropy import binary_entropy

__all__ = [
    "bsc_capacity",
    "bec_capacity",
    "awgn_capacity",
    "minimum_ebn0",
    "blahut_arimoto",
    "random_code_error_rate",
    "rate_distortion_binary",
    "rate_distortion_gaussian",
]


def _scalar_or_array(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def bsc_capacity(p):
    r"""Capacity :math:`C = 1 - h(p)` of the binary symmetric channel with crossover probability :math:`p`.

    Shannon (1948), Section 13: each bit is flipped independently with
    probability :math:`p`, and a uniform input achieves capacity.

    Parameters
    ----------
    p : float or array_like

    Returns
    -------
    float or ndarray
        Bits per channel use.

    Examples
    --------
    >>> bsc_capacity(0.0), bsc_capacity(0.5)
    (1.0, 0.0)
    >>> round(bsc_capacity(0.11), 3)
    0.5
    """
    return _scalar_or_array(1.0 - np.asarray(binary_entropy(p)))


def bec_capacity(epsilon):
    r"""Capacity :math:`C = 1 - \varepsilon` of the binary erasure channel.

    P. Elias, "Coding for Two Noisy Channels," Information Theory, Third
    London Symposium (1955), 61-76.

    Parameters
    ----------
    epsilon : float or array_like
        Erasure probability.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> bec_capacity(0.25)
    0.75
    """
    epsilon = np.asarray(epsilon, dtype=float)
    if np.any((epsilon < 0) | (epsilon > 1)):
        raise ValueError("epsilon must lie in [0, 1]")
    return _scalar_or_array(1.0 - epsilon)


def awgn_capacity(snr, bandwidth: float = 1.0):
    r"""Shannon-Hartley capacity :math:`C = B\log_2(1 + S/N)` of a band-limited Gaussian channel.

    C. E. Shannon, "Communication in the Presence of Noise," Proceedings
    of the IRE 37(1) (1949), 10-21, Theorem 2.

    Parameters
    ----------
    snr : float or array_like
        Signal-to-noise power ratio :math:`S/N` (linear, not dB).
    bandwidth : float
        :math:`B` in Hz; the default 1 gives bits/s/Hz.

    Returns
    -------
    float or ndarray
        Bits per second.

    Examples
    --------
    >>> awgn_capacity(1.0)
    1.0
    >>> round(awgn_capacity(10 ** (30 / 10), bandwidth=3000))  # a 30 dB telephone line
    29902
    """
    snr = np.asarray(snr, dtype=float)
    if np.any(snr < 0):
        raise ValueError("snr must be non-negative")
    return _scalar_or_array(bandwidth * np.log2(1.0 + snr))


def minimum_ebn0(spectral_efficiency):
    r"""Smallest energy per bit :math:`E_b/N_0 = (2^\eta - 1)/\eta` that supports :math:`\eta` bits/s/Hz.

    As :math:`\eta \to 0` this tends to :math:`\ln 2`, the Shannon limit
    of :math:`-1.59` dB below which no code communicates reliably.

    Parameters
    ----------
    spectral_efficiency : float or array_like
        :math:`\eta = C/B > 0`.

    Returns
    -------
    float or ndarray
        Linear ratio (``10 * log10`` for dB).

    Examples
    --------
    >>> minimum_ebn0(1.0)
    1.0
    >>> round(float(10 * np.log10(minimum_ebn0(1e-9))), 2)
    -1.59
    """
    eta = np.asarray(spectral_efficiency, dtype=float)
    if np.any(eta <= 0):
        raise ValueError("spectral_efficiency must be positive")
    return _scalar_or_array(np.expm1(eta * np.log(2.0)) / eta)


def blahut_arimoto(transition, tol: float = DEFAULT_ATOL, max_iter: int = DEFAULT_MAX_ITER) -> ChannelCapacityResult:
    r"""Capacity of a discrete memoryless channel by the Blahut-Arimoto algorithm.

    Alternates between the output distribution
    :math:`q(y) = \sum_x r(x)W(y\mid x)` and the input update
    :math:`r(x) \propto r(x)\exp D\bigl(W(\cdot\mid x)\,\|\,q\bigr)`.
    At every step :math:`\log\sum_x r(x)e^{D_x} \le C \le \max_x D_x`, and
    the iteration stops when the bounds meet within `tol`. R. Blahut,
    "Computation of Channel Capacity and Rate-Distortion Functions," IEEE
    Transactions on Information Theory 18(4) (1972), 460-473; S. Arimoto,
    "An Algorithm for Computing the Capacity of Arbitrary Discrete
    Memoryless Channels," ibid., 14-20.

    Parameters
    ----------
    transition : array_like, shape (m, n)
        ``transition[x, y]`` :math:`= W(y\mid x)`; rows sum to 1.
    tol : float
        Gap between the capacity bounds, in bits.
    max_iter : int

    Returns
    -------
    ChannelCapacityResult

    Examples
    --------
    >>> r = blahut_arimoto([[0.9, 0.1], [0.1, 0.9]])
    >>> round(r.capacity, 6), r.input_distribution.round(3).tolist()
    (0.531004, [0.5, 0.5])
    >>> round(blahut_arimoto([[1, 0], [0.5, 0.5]]).capacity, 4)  # the Z channel
    0.3219
    """
    W = np.asarray(transition, dtype=float)
    if W.ndim != 2 or np.any(W < 0) or not np.allclose(W.sum(axis=1), 1.0):
        raise ValueError("transition must be a 2-D array of non-negative rows summing to 1")
    r = np.full(W.shape[0], 1.0 / W.shape[0])
    lower, upper = [], []
    converged = False
    for _ in range(max_iter):
        q = r @ W
        D = rel_entr(W, q[np.newaxis, :]).sum(axis=1)
        c = np.exp(D)
        lower.append(np.log(r @ c) / np.log(2.0))
        upper.append(D.max() / np.log(2.0))
        r = r * c / (r @ c)
        if upper[-1] - lower[-1] < tol:
            converged = True
            break
    return ChannelCapacityResult(
        capacity=float(lower[-1]),
        input_distribution=r,
        iterations=len(lower),
        converged=converged,
        lower_bounds=np.array(lower),
        upper_bounds=np.array(upper),
    )


def random_code_error_rate(n: int, rate: float, p: float, trials: int = 1000, seed=None) -> float:
    r"""Block error rate of random codes with maximum-likelihood decoding on a binary symmetric channel.

    Shannon's proof of the noisy-channel coding theorem (1948, Theorem
    11) draws :math:`2^{nR}` codewords of length :math:`n` uniformly at
    random. Averaged over that ensemble, the error probability of
    minimum-Hamming-distance decoding tends to 0 as :math:`n` grows when
    :math:`R < C = 1 - h(p)`, and to 1 when :math:`R > C`. Each trial
    draws a fresh codebook, sends its first codeword, and breaks ties
    in distance at random.

    Parameters
    ----------
    n : int
        Block length.
    rate : float
        Code rate :math:`R`; the codebook has :math:`\mathrm{round}(2^{nR})`
        codewords.
    p : float
        Crossover probability.
    trials : int
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    float
        Estimated block error probability.

    Examples
    --------
    >>> random_code_error_rate(16, 0.25, p=0.0, trials=200, seed=0) < 0.05
    True
    """
    rng = np.random.default_rng(seed)
    M = max(round(2.0 ** (n * rate)), 2)
    errors = 0.0
    for _ in range(trials):
        codebook = rng.integers(0, 2, size=(M, n), dtype=np.uint8)
        received = codebook[0] ^ (rng.random(n) < p).astype(np.uint8)
        distances = np.count_nonzero(codebook != received, axis=1)
        d_sent = distances[0]
        if np.any(distances[1:] < d_sent):
            errors += 1.0
        else:
            ties = np.count_nonzero(distances[1:] == d_sent)
            errors += ties / (ties + 1)
    return float(errors / trials)


def rate_distortion_binary(D, p: float = 0.5):
    r"""Rate-distortion function :math:`R(D) = h(p) - h(D)` of a Bernoulli(:math:`p`) source under Hamming distortion.

    Zero for :math:`D \ge \min(p, 1-p)`. C. E. Shannon, "Coding Theorems
    for a Discrete Source with a Fidelity Criterion," IRE National
    Convention Record 7(4) (1959), 142-163.

    Parameters
    ----------
    D : float or array_like
        Allowed fraction of bit errors.
    p : float

    Returns
    -------
    float or ndarray
        Bits per source symbol.

    Examples
    --------
    >>> rate_distortion_binary(0.0), rate_distortion_binary(0.5)
    (1.0, 0.0)
    >>> round(rate_distortion_binary(0.11), 3)
    0.5
    """
    D = np.asarray(D, dtype=float)
    if np.any(D < 0):
        raise ValueError("D must be non-negative")
    Dc = np.minimum(D, min(p, 1.0 - p))
    return _scalar_or_array(binary_entropy(p) - np.asarray(binary_entropy(Dc)))


def rate_distortion_gaussian(D, variance: float = 1.0):
    r"""Rate-distortion function :math:`R(D) = \max\bigl(0, \tfrac12\log_2(\sigma^2/D)\bigr)` of a Gaussian source under squared error.

    Shannon (1959). Each extra bit per sample divides the achievable
    mean squared error by 4 (6.02 dB).

    Parameters
    ----------
    D : float or array_like
        Allowed mean squared error, positive.
    variance : float
        Source variance :math:`\sigma^2`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> rate_distortion_gaussian(0.25)
    1.0
    >>> rate_distortion_gaussian(2.0)
    0.0
    """
    D = np.asarray(D, dtype=float)
    if np.any(D <= 0):
        raise ValueError("D must be positive")
    return _scalar_or_array(np.maximum(0.0, 0.5 * np.log2(variance / D)))
