r"""The Laplace transform: numerical evaluation and numerical inversion.

The forward transform :math:`F(s) = \int_0^\infty f(t)\,e^{-st}\,dt` is a
single improper integral, computed with :func:`scipy.integrate.quad`.
Inversion is harder: the Bromwich integral
:math:`f(t) = \frac{1}{2\pi i}\int_{\gamma-i\infty}^{\gamma+i\infty}
F(s)\,e^{st}\,ds` is highly oscillatory along a vertical line, and
neither numpy nor scipy offers a routine for it, so two classical
algorithms are hand-rolled here:

- the fixed Talbot method, which deforms the Bromwich contour into a
  path that winds around the negative real axis, where :math:`e^{st}`
  decays and the trapezoidal rule converges rapidly. Needs :math:`F` at
  complex :math:`s`. See A. Talbot, "The accurate numerical inversion of
  Laplace transforms," J. Inst. Math. Appl. 23 (1979), 97-120, and J.
  Abate and P. P. Valkó, "Multi-precision Laplace transform inversion,"
  Int. J. Numer. Meth. Eng. 60 (2004), 979-993.
- the Gaver-Stehfest method, a Salzer-accelerated sequence of real-axis
  samples. Needs :math:`F` only at real :math:`s`, but loses digits to
  cancellation and suits smooth, non-oscillating :math:`f`. See H.
  Stehfest, "Algorithm 368: Numerical inversion of Laplace transforms,"
  Comm. ACM 13 (1970), 47-49.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import factorial, log

import numpy as np
from scipy import integrate

__all__ = ["laplace_transform", "inverse_laplace_talbot", "inverse_laplace_stehfest", "stehfest_coefficients"]


def laplace_transform(f: Callable[[float], float], s) -> np.ndarray:
    r"""The Laplace transform :math:`F(s) = \int_0^\infty f(t)\,e^{-st}\,dt`, by adaptive quadrature.

    Uses :func:`scipy.integrate.quad` on :math:`[0, \infty)`, integrating
    the real and imaginary parts separately for complex :math:`s`.
    :math:`\operatorname{Re} s` must exceed the abscissa of convergence
    of :math:`f`.

    Parameters
    ----------
    f : callable
        ``f(t) -> float`` for scalar ``t >= 0``.
    s : complex or array_like

    Returns
    -------
    ndarray
        Same shape as ``s``; real when every ``s`` is real.

    Examples
    --------
    >>> import numpy as np
    >>> F = laplace_transform(np.sin, [1.0, 2.0])  # 1 / (s^2 + 1)
    >>> np.allclose(F, [0.5, 0.2])
    True
    """
    s_arr = np.asarray(s)
    complex_input = np.iscomplexobj(s_arr)
    flat = s_arr.ravel()
    out = np.empty(flat.shape, dtype=np.complex128 if complex_input else float)
    for i, sk in enumerate(flat):
        re = integrate.quad(lambda t: f(t) * np.exp(-sk.real * t) * np.cos(sk.imag * t), 0.0, np.inf, limit=200)[0]
        if complex_input:
            im = integrate.quad(lambda t: -f(t) * np.exp(-sk.real * t) * np.sin(sk.imag * t), 0.0, np.inf, limit=200)[0]
            out[i] = complex(re, im)
        else:
            out[i] = re
    return out.reshape(s_arr.shape)


def inverse_laplace_talbot(F: Callable[[np.ndarray], np.ndarray], t, m: int = 32) -> np.ndarray:
    r"""Invert a Laplace transform numerically by the fixed Talbot method.

    With :math:`r = 2m/(5t)` and :math:`\theta_k = k\pi/m`, the contour
    :math:`s(\theta) = r\theta(\cot\theta + i)` gives (Abate & Valkó 2004,
    Eq. 36)

    .. math::

       f(t) \approx \frac{r}{m}\left[\tfrac12 F(r)e^{rt}
       + \sum_{k=1}^{m-1}\operatorname{Re}\Bigl(e^{t s(\theta_k)}F\bigl(s(\theta_k)\bigr)
       \bigl(1 + i\sigma(\theta_k)\bigr)\Bigr)\right],
       \quad \sigma(\theta) = \theta + (\theta\cot\theta - 1)\cot\theta .

    In double precision, accuracy saturates near :math:`10^{-10}` for
    :math:`m \approx 20`-:math:`40`; larger :math:`m` amplifies roundoff.
    Every singularity of :math:`F` must lie inside the contour, which
    crosses the imaginary axis at :math:`\pm i m\pi/(5t)`. Poles in the
    right half-plane are not supported, and a pole :math:`\alpha \pm
    i\beta` near the imaginary axis (an oscillating :math:`f`) needs
    :math:`t < m\pi/(5\beta)`, in practice with some margin: raise
    ``m`` to reach later times.

    Parameters
    ----------
    F : callable
        ``F(s)`` accepting a complex ndarray and returning an array of the same shape.
    t : float or array_like
        Times, all ``> 0``.
    m : int
        Number of contour nodes.

    Returns
    -------
    ndarray
        Same shape as ``t``.

    Examples
    --------
    >>> import numpy as np
    >>> f = inverse_laplace_talbot(lambda s: 1.0 / (s + 1.0), [0.5, 1.0, 2.0])
    >>> bool(np.allclose(f, np.exp(-np.array([0.5, 1.0, 2.0])), atol=1e-9))
    True
    """
    t_arr = np.asarray(t, dtype=float)
    if np.any(t_arr <= 0):
        raise ValueError("inverse_laplace_talbot requires t > 0")
    tt = t_arr.reshape(-1, 1)
    r = 2.0 * m / (5.0 * tt)
    theta = np.arange(1, m) * np.pi / m
    cot = 1.0 / np.tan(theta)
    s = r * theta * (cot + 1j)
    sigma = theta + (theta * cot - 1.0) * cot
    terms = np.real(np.exp(tt * s) * F(s) * (1.0 + 1j * sigma))
    first = 0.5 * np.real(F(r.astype(np.complex128))) * np.exp(r * tt)
    f = (r / m) * (first + terms.sum(axis=1, keepdims=True))
    return f.reshape(t_arr.shape)


def stehfest_coefficients(n: int) -> np.ndarray:
    r"""The Gaver-Stehfest weights :math:`V_1, \ldots, V_n` (``n`` even).

    .. math::

       V_k = (-1)^{k + n/2} \sum_{j=\lfloor (k+1)/2 \rfloor}^{\min(k, n/2)}
       \frac{j^{n/2}\,(2j)!}{(n/2 - j)!\,j!\,(j-1)!\,(k-j)!\,(2j-k)!}

    (Stehfest 1970). Computed in exact integer arithmetic, then converted
    to float; the weights sum to zero and grow like :math:`10^{n/2}`,
    which is what limits the method's accuracy in double precision.

    Parameters
    ----------
    n : int
        Even number of terms.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> stehfest_coefficients(4)
    array([ -2.,  26., -48.,  24.])
    """
    if n <= 0 or n % 2:
        raise ValueError(f"n must be a positive even integer, got {n}")
    half = n // 2
    v = np.empty(n)
    for k in range(1, n + 1):
        total = Fraction(0)
        for j in range((k + 1) // 2, min(k, half) + 1):
            num = j**half * factorial(2 * j)
            den = factorial(half - j) * factorial(j) * factorial(j - 1) * factorial(k - j) * factorial(2 * j - k)
            total += Fraction(num, den)
        v[k - 1] = float((-1) ** (k + half) * total)
    return v


def inverse_laplace_stehfest(F: Callable[[np.ndarray], np.ndarray], t, n: int = 14) -> np.ndarray:
    r"""Invert a Laplace transform numerically by the Gaver-Stehfest method.

    .. math::

       f(t) \approx \frac{\ln 2}{t}\sum_{k=1}^{n} V_k\,F\!\left(\frac{k\ln 2}{t}\right)

    with :func:`stehfest_coefficients` :math:`V_k`. Only real :math:`s`
    are needed. In double precision the error is smallest near
    :math:`n = 12`-:math:`16` (about 5-7 digits for smooth :math:`f`);
    it cannot resolve oscillating or discontinuous :math:`f`.

    Parameters
    ----------
    F : callable
        ``F(s)`` accepting a real ndarray and returning an array of the same shape.
    t : float or array_like
        Times, all ``> 0``.
    n : int
        Even number of terms.

    Returns
    -------
    ndarray
        Same shape as ``t``.

    Examples
    --------
    >>> import numpy as np
    >>> f = inverse_laplace_stehfest(lambda s: 1.0 / (s + 1.0), [0.5, 1.0, 2.0])
    >>> bool(np.allclose(f, np.exp(-np.array([0.5, 1.0, 2.0])), atol=1e-5))
    True
    """
    t_arr = np.asarray(t, dtype=float)
    if np.any(t_arr <= 0):
        raise ValueError("inverse_laplace_stehfest requires t > 0")
    v = stehfest_coefficients(n)
    tt = t_arr.reshape(-1, 1)
    s = np.arange(1, n + 1) * log(2.0) / tt
    f = (log(2.0) / tt[:, 0]) * (F(s) @ v)
    return f.reshape(t_arr.shape)
