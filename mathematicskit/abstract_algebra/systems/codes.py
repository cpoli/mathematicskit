r"""Reed-Solomon codes over a prime field :math:`\mathrm{GF}(p)`, in their
original evaluation form.

A message of :math:`k` symbols is read as the coefficients of a
polynomial of degree below :math:`k`; the codeword is that polynomial's
values at :math:`n` distinct field elements. Any :math:`k` surviving
values determine the polynomial by Lagrange interpolation, so the code
recovers from up to :math:`n-k` erasures. No numpy/scipy equivalent
(exact modular arithmetic). See I. S. Reed and G. Solomon, "Polynomial
Codes over Certain Finite Fields," Journal of the Society for Industrial
and Applied Mathematics 8(2) (1960), 300-304.
"""

from __future__ import annotations

__all__ = ["rs_encode", "rs_decode_erasures"]


def _check_prime_field(p: int, n: int, k: int) -> None:
    if p < 2 or any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
        raise ValueError(f"p = {p} must be prime")
    if not 0 < k <= n <= p:
        raise ValueError("need 0 < k <= n <= p")


def rs_encode(message, n: int, p: int) -> list:
    r"""Encode `message` as the values :math:`m(0), m(1), \dots, m(n-1)` of its polynomial over :math:`\mathrm{GF}(p)`.

    Parameters
    ----------
    message : sequence of int
        The :math:`k` message symbols, each in ``range(p)``; ``message[i]``
        is the coefficient of :math:`x^i`.
    n : int
        Codeword length, :math:`k \le n \le p`.
    p : int
        A prime field size.

    Returns
    -------
    list of int

    Examples
    --------
    >>> rs_encode([3, 1], n=5, p=7)  # m(x) = 3 + x at x = 0..4
    [3, 4, 5, 6, 0]
    """
    message = [int(c) % p for c in message]
    _check_prime_field(p, n, len(message))
    codeword = []
    for x in range(n):
        value = 0
        for c in reversed(message):
            value = (value * x + c) % p
        codeword.append(value)
    return codeword


def rs_decode_erasures(received, k: int, p: int) -> list:
    r"""Recover a :math:`k`-symbol message from a codeword with erased symbols.

    Lagrange-interpolates the unique polynomial of degree below
    :math:`k` through any :math:`k` surviving positions, working exactly
    in :math:`\mathrm{GF}(p)`.

    Parameters
    ----------
    received : sequence
        The codeword, with ``None`` at each erased position.
    k : int
        Message length.
    p : int
        The prime field size used to encode.

    Returns
    -------
    list of int
        The message symbols (polynomial coefficients, lowest degree first).

    Examples
    --------
    >>> codeword = rs_encode([2, 5, 1], n=6, p=11)
    >>> damaged = [None, codeword[1], None, codeword[3], codeword[4], None]
    >>> rs_decode_erasures(damaged, k=3, p=11)
    [2, 5, 1]
    """
    _check_prime_field(p, len(received), k)
    points = [(x, int(y) % p) for x, y in enumerate(received) if y is not None][:k]
    if len(points) < k:
        raise ValueError(f"only {len(points)} symbols survive; at least k = {k} are needed")
    coeffs = [0] * k
    for i, (xi, yi) in enumerate(points):
        basis = [1]  # the Lagrange basis polynomial prod_{j != i} (x - xj) / (xi - xj)
        denominator = 1
        for j, (xj, _) in enumerate(points):
            if j == i:
                continue
            basis = [((basis[t - 1] if t > 0 else 0) - xj * (basis[t] if t < len(basis) else 0)) % p for t in range(len(basis) + 1)]
            denominator = denominator * (xi - xj) % p
        scale = yi * pow(denominator, p - 2, p) % p
        for t, b in enumerate(basis):
            coeffs[t] = (coeffs[t] + scale * b) % p
    return coeffs
