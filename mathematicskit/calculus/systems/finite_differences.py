r"""Finite-difference derivatives (forward/backward/central) and Richardson
extrapolation for higher accuracy.

See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 4.1
("Numerical Differentiation") for the difference formulas and their
truncation-error orders, and Ch. 4.2 ("Richardson's Extrapolation") for
combining two lower-order estimates into a higher-order one.
"""

from __future__ import annotations

from typing import Callable

from mathematicskit.calculus.core.base import DerivativeResult

__all__ = ["forward_difference", "backward_difference", "central_difference", "richardson_extrapolation"]


def forward_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    r"""Forward-difference derivative estimate, :math:`O(h)`.

    :math:`f'(x) \approx \dfrac{f(x+h) - f(x)}{h}`. See Burden & Faires,
    *Numerical Analysis*, 10th ed., Ch. 4.1.

    Parameters
    ----------
    f : callable
    x : float
    h : float
        Step size.

    Returns
    -------
    float

    Examples
    --------
    >>> round(forward_difference(lambda t: t**2, 3.0, h=1e-4), 3)
    6.0
    """
    return (f(x + h) - f(x)) / h


def backward_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    r"""Backward-difference derivative estimate, :math:`O(h)`.

    :math:`f'(x) \approx \dfrac{f(x) - f(x-h)}{h}`.

    Parameters
    ----------
    f : callable
    x : float
    h : float

    Returns
    -------
    float

    Examples
    --------
    >>> round(backward_difference(lambda t: t**2, 3.0, h=1e-4), 3)
    6.0
    """
    return (f(x) - f(x - h)) / h


def central_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    r"""Central-difference derivative estimate, :math:`O(h^2)`.

    :math:`f'(x) \approx \dfrac{f(x+h) - f(x-h)}{2h}` -- one order of
    accuracy better than forward/backward differences at the same cost
    (2 evaluations), since the :math:`O(h)` error terms in the underlying
    Taylor expansions cancel. See Burden & Faires, *Numerical Analysis*,
    10th ed., Ch. 4.1, eq. (4.4).

    Parameters
    ----------
    f : callable
    x : float
    h : float

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> round(float(central_difference(np.sin, 0.0, h=1e-4)), 8)
    1.0
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def richardson_extrapolation(f: Callable[[float], float], x: float, h: float = 1e-2, levels: int = 4) -> DerivativeResult:
    r"""Richardson-extrapolate central differences to arbitrarily high order.

    Central differences have an error series in even powers of ``h``:
    :math:`D(h) = f'(x) + c_1 h^2 + c_2 h^4 + \dots`. Combining
    :math:`D(h)` and :math:`D(h/2)` as
    :math:`\dfrac{4 D(h/2) - D(h)}{3}` cancels the :math:`h^2` term,
    leaving :math:`O(h^4)`; repeating with successively halved step sizes
    in a triangular (Neville-style) table cancels one more error order
    each level. See Burden & Faires, *Numerical Analysis*, 10th ed.,
    Ch. 4.2, and the closely related idea in Ch. 4.5 (Romberg
    integration).

    Parameters
    ----------
    f : callable
    x : float
    h : float
        Coarsest step size; halved at each level.
    levels : int
        Number of extrapolation levels (``levels=1`` is plain central
        difference at step `h`).

    Returns
    -------
    DerivativeResult
        ``error_estimate`` is the change between the last two
        extrapolated values, a practical (not rigorous) error indicator.

    Examples
    --------
    >>> import numpy as np
    >>> result = richardson_extrapolation(np.sin, 0.0, h=0.2, levels=5)
    >>> round(float(result.value), 12)
    1.0
    """
    table = [[central_difference(f, x, h / (2**i)) for i in range(levels)]]
    for level in range(1, levels):
        row = []
        prev = table[level - 1]
        factor = 4.0**level
        for i in range(len(prev) - 1):
            row.append((factor * prev[i + 1] - prev[i]) / (factor - 1.0))
        table.append(row)
    best = float(table[-1][0])
    error_estimate = float(abs(table[-1][0] - table[-2][0])) if levels > 1 else 0.0
    return DerivativeResult(value=best, step=h / (2 ** (levels - 1)), method="richardson", error_estimate=error_estimate)
