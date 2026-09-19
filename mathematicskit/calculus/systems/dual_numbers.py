r"""Forward-mode automatic differentiation via dual numbers.

A dual number :math:`a + b\varepsilon` with :math:`\varepsilon^2 = 0`
propagates a function's value *and* its exact derivative together
through ordinary arithmetic: substituting :math:`x + \varepsilon` (i.e.
:math:`dx/dx = 1`) into any function built from ``+``, ``-``, ``*``,
``/``, and elementary functions and expanding via
:math:`f(a+b\varepsilon) = f(a) + b f'(a)\varepsilon` (Taylor's theorem,
truncated after the linear term since :math:`\varepsilon^2=0`) computes
:math:`f(x)` and :math:`f'(x)` in one pass with no finite-difference
truncation error. See Griewank & Walther, *Evaluating Derivatives*, 2nd
ed., Ch. 3, and Clifford (1873) for the original dual-number algebra.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = ["Dual", "derivative"]


class Dual:
    r"""A dual number :math:`\text{real} + \text{dual}\,\varepsilon`.

    Parameters
    ----------
    real : float
        The value part.
    dual : float
        The derivative (tangent) part.

    Examples
    --------
    >>> x = Dual(3.0, 1.0)  # seed dx/dx = 1
    >>> y = x * x + 2.0 * x  # f(x) = x^2 + 2x, f'(x) = 2x + 2
    >>> y.real, y.dual
    (15.0, 8.0)
    """

    __slots__ = ("real", "dual")

    def __init__(self, real: float, dual: float = 0.0):
        self.real = float(real)
        self.dual = float(dual)

    def __repr__(self):
        return f"Dual({self.real!r}, {self.dual!r})"

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Dual) else Dual(other, 0.0)

    def __add__(self, other):
        other = self._coerce(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        # Product rule: (u v)' = u'v + u v'
        return Dual(self.real * other.real, self.dual * other.real + self.real * other.dual)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        # Quotient rule: (u/v)' = (u'v - u v') / v^2
        return Dual(self.real / other.real, (self.dual * other.real - self.real * other.dual) / (other.real**2))

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __pow__(self, p: float):
        # Power rule: (u^p)' = p u^(p-1) u'
        return Dual(self.real**p, p * self.real ** (p - 1) * self.dual)

    def sin(self):
        return Dual(math.sin(self.real), math.cos(self.real) * self.dual)

    def cos(self):
        return Dual(math.cos(self.real), -math.sin(self.real) * self.dual)

    def exp(self):
        e = math.exp(self.real)
        return Dual(e, e * self.dual)

    def log(self):
        return Dual(math.log(self.real), self.dual / self.real)

    def sqrt(self):
        s = math.sqrt(self.real)
        return Dual(s, self.dual / (2.0 * s))


def derivative(f: Callable[[Dual], Dual], x: float) -> float:
    r"""Exact derivative of ``f`` at ``x`` via forward-mode dual numbers.

    Parameters
    ----------
    f : callable
        A function built from ``Dual`` arithmetic and methods
        (``+``, ``-``, ``*``, ``/``, ``**``, ``.sin()``, ``.cos()``,
        ``.exp()``, ``.log()``, ``.sqrt()``), taking and returning
        ``Dual``.
    x : float
        Point at which to differentiate.

    Returns
    -------
    float
        :math:`f'(x)`, exact up to floating-point roundoff (no
        truncation error, unlike finite differences).

    Examples
    --------
    >>> f = lambda x: (x * x).sin()
    >>> round(derivative(f, 1.0), 10) == round(2.0 * 1.0 * math.cos(1.0 * 1.0), 10)
    True
    """
    return f(Dual(x, 1.0)).dual
