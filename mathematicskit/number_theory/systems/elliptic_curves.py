r"""Elliptic curves over :math:`\mathrm{GF}(p)` and Lenstra's elliptic-curve factorization.

The points of :math:`E: y^2 = x^3 + ax + b` over a field, together with
a point at infinity :math:`\mathcal O`, form an abelian group under the
chord-and-tangent law: the line through :math:`P` and :math:`Q` meets
the curve in a third point, and :math:`P + Q` is its reflection in the
:math:`x`-axis. Over :math:`\mathrm{GF}(p)` the group is finite, and
Hasse (1933) proved that its order satisfies
:math:`|\#E - (p + 1)| \leq 2\sqrt p`.

Lenstra (1987) turned this into a factoring method. Doing the
arithmetic modulo a composite :math:`n` is secretly doing it modulo
each prime :math:`p \mid n` at once; multiplying a point by
:math:`k = \operatorname{lcm}(1, \dots, B)` sends it to :math:`\mathcal O`
modulo :math:`p` whenever :math:`\#E(\mathrm{GF}(p))` is
:math:`B`-smooth, and that shows up as a slope denominator that shares
the factor :math:`p` with :math:`n`. Unlike Pollard's :math:`p - 1`
method, a failure is cured by trying another random curve, with
another group order. Hand-rolled: exact integer arithmetic. See H. W.
Lenstra Jr., "Factoring Integers with Elliptic Curves," Annals of
Mathematics 126(3) (1987), 649-673, and Washington, *Elliptic Curves:
Number Theory and Cryptography*, 2nd ed. (2008), Ch. 2, 4 and 7.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from mathematicskit.number_theory.core.base import ECMResult
from mathematicskit.number_theory.systems.primality import is_prime_miller_rabin, sieve_of_eratosthenes

__all__ = ["EllipticCurve", "lenstra_ecm"]

Point = Optional[tuple]  # (x, y), or None for the point at infinity


class EllipticCurve:
    r"""The curve :math:`y^2 = x^3 + ax + b` over :math:`\mathrm{GF}(p)`, with its group law.

    Points are tuples ``(x, y)`` of residues; ``None`` is the point at
    infinity :math:`\mathcal O`, the group's identity.

    Parameters
    ----------
    a, b : int
    p : int
        A prime ``p > 3``.

    Examples
    --------
    >>> E = EllipticCurve(2, 3, 97)
    >>> P = (3, 6)
    >>> E.is_on_curve(P)
    True
    >>> E.add(P, P), E.multiply(2, P)
    ((80, 10), (80, 10))
    >>> E.multiply(E.point_order(P), P) is None
    True
    """

    def __init__(self, a: int, b: int, p: int):
        if p <= 3:
            raise ValueError("p must be a prime greater than 3")
        self.a, self.b, self.p = a % p, b % p, p
        if (4 * self.a**3 + 27 * self.b**2) % p == 0:
            raise ValueError("singular curve: 4a^3 + 27b^2 = 0 (mod p)")

    def is_on_curve(self, point: Point) -> bool:
        """Whether ``point`` satisfies the curve equation (``None`` always does)."""
        if point is None:
            return True
        x, y = point
        return (y * y - x**3 - self.a * x - self.b) % self.p == 0

    def negate(self, point: Point) -> Point:
        r""":math:`-P = (x, -y)`."""
        return None if point is None else (point[0], (-point[1]) % self.p)

    def add(self, p1: Point, p2: Point) -> Point:
        r"""The group law :math:`P + Q`.

        With slope :math:`\lambda = (y_2 - y_1)/(x_2 - x_1)`, or
        :math:`\lambda = (3x_1^2 + a)/(2y_1)` when doubling,
        :math:`x_3 = \lambda^2 - x_1 - x_2` and
        :math:`y_3 = \lambda(x_1 - x_3) - y_1`.
        """
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        (x1, y1), (x2, y2) = p1, p2
        p = self.p
        if x1 == x2 and (y1 + y2) % p == 0:
            return None
        if x1 == x2:
            slope = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, p) % p
        else:
            slope = (y2 - y1) * pow(x2 - x1, -1, p) % p
        x3 = (slope * slope - x1 - x2) % p
        return (x3, (slope * (x1 - x3) - y1) % p)

    def multiply(self, k: int, point: Point) -> Point:
        """:math:`kP` by double-and-add, in :math:`O(\\log k)` group operations."""
        if k < 0:
            return self.multiply(-k, self.negate(point))
        result: Point = None
        addend = point
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def points(self) -> list:
        """Every affine point of the curve (the point at infinity excluded), sorted.

        :math:`O(p)`: tabulates the square roots of every residue once.
        """
        p = self.p
        roots: dict = {}
        for y in range(p):
            roots.setdefault(y * y % p, []).append(y)
        return [(x, y) for x in range(p) for y in roots.get((x**3 + self.a * x + self.b) % p, [])]

    def order(self) -> int:
        r""":math:`\#E(\mathrm{GF}(p))`, the number of points including :math:`\mathcal O`."""
        return len(self.points()) + 1

    def point_order(self, point: Point) -> int:
        """The order of ``point``: the least ``k > 0`` with ``k * point`` at infinity."""
        k, q = 1, point
        while q is not None:
            q = self.add(q, point)
            k += 1
        return k


class _FactorFound(Exception):
    def __init__(self, factor: int):
        self.factor = factor


def _inverse_or_factor(d: int, n: int) -> int:
    g = math.gcd(d, n)
    if g != 1:
        raise _FactorFound(g)
    return pow(d, -1, n)


def _add_mod_n(p1: Point, p2: Point, a: int, n: int) -> Point:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    (x1, y1), (x2, y2) = p1, p2
    if x1 == x2 and (y1 + y2) % n == 0:
        return None
    if x1 == x2:
        slope = (3 * x1 * x1 + a) * _inverse_or_factor(2 * y1 % n, n) % n
    else:
        slope = (y2 - y1) * _inverse_or_factor((x2 - x1) % n, n) % n
    x3 = (slope * slope - x1 - x2) % n
    return (x3, (slope * (x1 - x3) - y1) % n)


def _multiply_mod_n(k: int, point: Point, a: int, n: int) -> Point:
    result: Point = None
    while k:
        if k & 1:
            result = _add_mod_n(result, point, a, n)
        point = _add_mod_n(point, point, a, n)
        k >>= 1
    return result


def lenstra_ecm(n: int, b1: int = 2000, max_curves: int = 500, seed: int = 0) -> ECMResult:
    r"""Find a nontrivial factor of composite ``n`` with Lenstra's elliptic-curve method.

    For each random curve and point modulo ``n``, computes
    :math:`kP` with :math:`k = \prod_{q \leq B_1} q^{\lfloor \log_q B_1 \rfloor}`,
    one prime power at a time. A slope denominator not invertible modulo
    ``n`` reveals :math:`\gcd(d, n)`; if that gcd is ``n`` itself, the
    curve is discarded. The expected running time depends on the size of
    the *smallest* prime factor :math:`p`, as
    :math:`\exp\bigl((\sqrt2 + o(1))\sqrt{\log p \log\log p}\bigr)`.

    Parameters
    ----------
    n : int
        Composite, ``n >= 4``. An even ``n`` returns the factor 2
        immediately.
    b1 : int
        Smoothness bound :math:`B_1`.
    max_curves : int
    seed : int

    Returns
    -------
    ECMResult

    Examples
    --------
    >>> result = lenstra_ecm(455839)
    >>> sorted([result.factor, result.cofactor])
    [599, 761]
    >>> result = lenstra_ecm(2**64 + 1)
    >>> sorted([result.factor, result.cofactor])
    [274177, 67280421310721]
    """
    if n < 4:
        raise ValueError("n must be a composite integer >= 4")
    if n % 2 == 0:
        return ECMResult(factor=2, cofactor=n // 2, curves=0, curve=())
    if is_prime_miller_rabin(n):
        raise ValueError(f"{n} is prime")
    prime_powers = []
    for q in sieve_of_eratosthenes(b1):
        q = int(q)
        qk = q
        while qk * q <= b1:
            qk *= q
        prime_powers.append(qk)
    rng = random.Random(seed)
    for curve_number in range(1, max_curves + 1):
        a, x0, y0 = rng.randrange(n), rng.randrange(n), rng.randrange(n)
        b = (y0 * y0 - x0**3 - a * x0) % n
        g = math.gcd(4 * a**3 + 27 * b * b, n)
        if g == n:
            continue
        if g > 1:
            return ECMResult(factor=g, cofactor=n // g, curves=curve_number, curve=(a, b, x0, y0))
        point: Point = (x0, y0)
        try:
            for qk in prime_powers:
                point = _multiply_mod_n(qk, point, a, n)
                if point is None:
                    break
        except _FactorFound as found:
            if found.factor != n:
                return ECMResult(factor=found.factor, cofactor=n // found.factor, curves=curve_number, curve=(a, b, x0, y0))
    raise RuntimeError(f"no factor of {n} found with {max_curves} curves and B1 = {b1}")
