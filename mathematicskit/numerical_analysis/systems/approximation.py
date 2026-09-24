"""Uniform and rational approximation of functions: Bernstein polynomials
(a constructive Weierstrass theorem), Padé approximants, and the Remez
exchange algorithm for best (minimax) polynomial approximation.

Bernstein weights come from :func:`scipy.stats.binom`; Padé coefficients
from one :func:`numpy.linalg.solve` (``scipy.interpolate.pade`` is
deprecated); the Remez iteration has no numpy/scipy equivalent for
general functions (:func:`scipy.signal.remez` designs FIR filters), so its
exchange loop is written out, with each levelled-error system solved by
:func:`numpy.linalg.solve` in the Chebyshev basis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import minimize_scalar
from scipy.stats import binom

from mathematicskit.constants import DEFAULT_MAX_ITER
from mathematicskit.numerical_analysis.core.base import MinimaxResult

__all__ = ["bernstein_polynomial", "PadeApproximant", "remez_minimax"]


def bernstein_polynomial(f: Callable, n: int, x):
    r"""Evaluate the degree-``n`` Bernstein polynomial of ``f`` on :math:`[0, 1]`.

    .. math::

        (B_n f)(x) = \sum_{k=0}^{n} f\!\left(\tfrac{k}{n}\right)
        \binom{n}{k} x^k (1 - x)^{n-k}

    The weights are the binomial probabilities
    :math:`P(K = k)` for :math:`K \sim \mathrm{Bin}(n, x)`, so
    :math:`B_n f(x) = \mathbb{E}[f(K/n)]`, and the law of large numbers
    forces :math:`B_n f \to f` uniformly for every continuous ``f``: a
    constructive proof of the Weierstrass approximation theorem (S.
    Bernstein, Comm. Soc. Math. Kharkov (2) 13 (1912), 1-2). Convergence
    is slow: :math:`B_n(x^2) = x^2 + x(1 - x)/n`.

    Parameters
    ----------
    f : callable
        Vectorized function on :math:`[0, 1]`.
    n : int
        Degree, ``n >= 1``.
    x : float or array-like
        Evaluation point(s) in :math:`[0, 1]`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(bernstein_polynomial(lambda t: t**2, 4, 0.5), 10)  # 0.25 + 0.25/4
    0.3125
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    x = np.asarray(x, dtype=np.float64)
    k = np.arange(n + 1)
    fk = np.asarray(f(k / n), dtype=np.float64) * np.ones(n + 1)
    weights = binom.pmf(k, n, x[..., None])
    out = weights @ fk
    return float(out) if x.ndim == 0 else out


class PadeApproximant:
    r"""Padé approximant :math:`[m/n]` of a power series: the rational
    function :math:`p(x)/q(x)`, :math:`\deg p \le m`, :math:`\deg q \le n`,
    :math:`q(0) = 1`, whose Taylor series agrees with :math:`f` through
    :math:`x^{m+n}`.

    With Taylor coefficients :math:`c_i` (and :math:`c_i = 0` for
    :math:`i < 0`), the denominator solves the :math:`n \times n` linear
    system :math:`\sum_{j=0}^{n} q_j c_{m+k-j} = 0`, :math:`k = 1, \ldots, n`,
    and then :math:`p_i = \sum_{j=0}^{\min(i, n)} q_j c_{i-j}`. See H. Padé,
    "Sur la représentation approchée d'une fonction par des fractions
    rationnelles," Annales scientifiques de l'École Normale Supérieure (3)
    9 (1892), supplement, 3-93; G. A. Baker and P. Graves-Morris, *Padé
    Approximants*, 2nd ed. (Cambridge University Press, 1996), Ch. 1.

    Parameters
    ----------
    taylor_coefficients : array-like
        :math:`c_0, c_1, \ldots`, lowest order first; at least
        ``m + n + 1`` of them.
    m, n : int
        Numerator and denominator degrees.

    Attributes
    ----------
    numerator, denominator : ndarray
        Coefficients of :math:`p` and :math:`q`, lowest order first.

    Examples
    --------
    >>> import math
    >>> c = [1.0 / math.factorial(k) for k in range(3)]
    >>> r = PadeApproximant(c, 1, 1)  # exp(x) ~ (1 + x/2) / (1 - x/2)
    >>> r.numerator.tolist(), r.denominator.tolist()
    ([1.0, 0.5], [1.0, -0.5])
    >>> round(r(1.0), 10)
    3.0
    """

    def __init__(self, taylor_coefficients, m: int, n: int):
        c = np.asarray(taylor_coefficients, dtype=np.float64)
        if m < 0 or n < 0:
            raise ValueError("m and n must be non-negative")
        if c.shape[0] < m + n + 1:
            raise ValueError(f"need at least m + n + 1 = {m + n + 1} Taylor coefficients, got {c.shape[0]}")
        self.m, self.n = int(m), int(n)

        def coef(i):
            return c[i] if i >= 0 else 0.0

        q = np.ones(n + 1)
        if n > 0:
            A = np.array([[coef(m + k - j) for j in range(1, n + 1)] for k in range(1, n + 1)])
            rhs = -np.array([coef(m + k) for k in range(1, n + 1)])
            q[1:] = np.linalg.solve(A, rhs)
        p = np.array([sum(q[j] * coef(i - j) for j in range(min(i, n) + 1)) for i in range(m + 1)])
        self.numerator = p
        self.denominator = q

    def evaluate(self, x):
        """Evaluate :math:`p(x)/q(x)`."""
        x = np.asarray(x, dtype=np.float64)
        out = np.polynomial.polynomial.polyval(x, self.numerator) / np.polynomial.polynomial.polyval(x, self.denominator)
        return float(out) if x.ndim == 0 else out

    def __call__(self, x):
        return self.evaluate(x)


def _alternating_extrema(err: np.ndarray) -> np.ndarray:
    """Index of the largest-|err| point in each maximal run of constant sign."""
    sign = np.sign(err)
    sign[sign == 0] = 1
    breaks = np.flatnonzero(np.diff(sign)) + 1
    runs = np.split(np.arange(err.shape[0]), breaks)
    return np.array([run[np.argmax(np.abs(err[run]))] for run in runs])


def remez_minimax(
    f: Callable,
    degree: int,
    a: float = -1.0,
    b: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = DEFAULT_MAX_ITER,
    n_grid: int = 20001,
) -> MinimaxResult:
    r"""Best uniform (minimax) polynomial approximation by the Remez exchange algorithm.

    By Chebyshev's equioscillation theorem, :math:`p^*` of degree
    :math:`\le n` minimizes :math:`\max_{[a,b]} |f - p|` if and only if the
    error attains its maximum magnitude with alternating sign at
    :math:`n + 2` points. Starting from the Chebyshev extrema, each step
    solves the linear system

    .. math::

        p(x_i) + (-1)^i E = f(x_i), \qquad i = 0, \ldots, n + 1,

    for :math:`p` and the levelled error :math:`E`, then moves the
    reference to the alternating extrema of the new error curve (located
    on a fine grid, then polished with :func:`scipy.optimize.minimize_scalar`), until they are equal to within `tol`. See E. Ya. Remez,
    "Sur la détermination des polynômes d'approximation de degré donnée,"
    Comm. Soc. Math. Kharkov (4) 10 (1934), 41-63; L. N. Trefethen,
    *Approximation Theory and Approximation Practice* (SIAM, 2013), Ch. 10.

    Parameters
    ----------
    f : callable
        Vectorized continuous function on :math:`[a, b]`.
    degree : int
        Polynomial degree :math:`n`.
    a, b : float
        Interval endpoints.
    tol : float
        Stop when :math:`(\max|e| - \min_i |e(x_i)|) / \max|e| < ` `tol`.
    max_iter : int
        Maximum number of exchange steps.
    n_grid : int
        Size of the grid used to locate error extrema.

    Returns
    -------
    MinimaxResult

    Examples
    --------
    >>> # Best cubic to x^4 on [-1, 1] is x^2 - 1/8, error 1/8 (= 2^{-3}).
    >>> res = remez_minimax(lambda x: x**4, 3)
    >>> bool(np.allclose(res.coefficients, [0.0, 1.0, 0.0, -0.125]))
    True
    >>> round(res.max_error, 8)
    0.125
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if not b > a:
        raise ValueError("need a < b")
    n = int(degree)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    grid_t = np.cos(np.linspace(np.pi, 0.0, n_grid))  # Chebyshev-clustered grid on [-1, 1]
    f_grid = np.asarray(f(mid + half * grid_t), dtype=np.float64)
    ref_t = np.cos(np.pi * np.arange(n + 1, -1, -1) / (n + 1))
    # Nudge the symmetric start off-centre: for an even f and a symmetric
    # reference the levelled error is forced to zero, stalling the exchange.
    ref_t = ref_t + 0.05 * (1.0 - ref_t**2)
    signs = (-1.0) ** np.arange(n + 2)

    def refine(cheb, idx):
        """Polish grid extrema of the error by a bounded 1-D search between grid neighbours."""
        err_fn = lambda t: float(np.asarray(f(mid + half * t))) - float(C.chebval(t, cheb))
        t_out, e_out = [], []
        for i in idx:
            t0, e0 = grid_t[i], err_fn(grid_t[i])
            lo_t, hi_t = grid_t[max(i - 1, 0)], grid_t[min(i + 1, n_grid - 1)]
            opt = minimize_scalar(lambda t: -abs(err_fn(t)), bounds=(lo_t, hi_t), method="bounded", options={"xatol": 1e-15})
            if -opt.fun > abs(e0):
                t0, e0 = float(opt.x), err_fn(float(opt.x))
            t_out.append(t0)
            e_out.append(e0)
        return np.array(t_out), np.array(e_out)

    converged = False
    iterations = 0
    cheb = np.zeros(n + 1)
    max_err = np.inf
    while iterations < max_iter:
        iterations += 1
        A = np.column_stack([C.chebvander(ref_t, n), signs])
        sol = np.linalg.solve(A, np.asarray(f(mid + half * ref_t), dtype=np.float64))
        cheb = sol[:-1]
        err = f_grid - C.chebval(grid_t, cheb)
        levelled = abs(float(sol[-1]))
        ext = _alternating_extrema(err)
        if ext.shape[0] < n + 2:
            max_err = float(np.max(np.abs(err)))
            break
        # Choose the window of n + 2 consecutive alternating extrema that
        # contains the global maximum and has the largest minimum |error|.
        i_max = int(np.argmax(np.abs(err[ext])))
        lo = max(0, i_max - (n + 1))
        hi = min(i_max, ext.shape[0] - (n + 2))
        best = max(range(lo, hi + 1), key=lambda s: np.min(np.abs(err[ext[s : s + n + 2]])))
        ref_t, ref_err = refine(cheb, ext[best : best + n + 2])
        signs = np.sign(ref_err)
        max_err = float(np.max(np.abs(ref_err)))
        if max_err == 0.0 or (max_err - levelled) / max_err < tol:
            converged = True
            break

    # Convert from the Chebyshev basis on [a, b] to the power basis in x.
    power = np.polynomial.Chebyshev(cheb, domain=[a, b]).convert(kind=np.polynomial.Polynomial).coef
    power = np.pad(power, (0, n + 1 - power.shape[0]))
    return MinimaxResult(coefficients=power[::-1], max_error=max_err, reference=mid + half * ref_t, iterations=iterations, converged=converged)
