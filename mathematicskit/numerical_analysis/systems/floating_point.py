r"""Floating-point arithmetic: IEEE 754 encodings, machine epsilon, the
spacing of floating-point numbers, and catastrophic cancellation.

A binary floating-point number is :math:`\pm m \cdot 2^{e - t}` with a
:math:`t`-bit integer significand :math:`m` and a bounded exponent
:math:`e`. IEEE 754 (1985) fixed the bit layout, the rounding rule
(round to nearest, ties to even), and the special values (signed zero,
subnormals, infinities, NaN), so the same program rounds identically on
every conforming machine. Every operation is then exact up to one
rounding: :math:`\mathrm{fl}(x \circ y) = (x \circ y)(1 + \delta)`, with
:math:`|\delta| \le u = \varepsilon/2`, the unit roundoff. Subtracting
two nearly equal *rounded* numbers is exact but exposes their earlier
rounding errors, which can wipe out most of the significant digits
("catastrophic cancellation"); the cure is to rewrite the formula so the
subtraction never happens.

See N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
ed. (SIAM, 2002), Ch. 1-2; D. Goldberg, "What Every Computer Scientist
Should Know About Floating-Point Arithmetic," ACM Computing Surveys 23
(1991), 5-48; and IEEE Std 754-2019. Machine epsilon is found by the
halving loop that textbooks use to introduce it (cross-checked against
:class:`numpy.finfo` in the tests); :func:`ulp` wraps
:func:`numpy.spacing`.
"""

from __future__ import annotations

import math

import numpy as np

from mathematicskit.numerical_analysis.core.base import FloatBits

__all__ = ["machine_epsilon", "float_bits", "ulp", "toy_float_system", "quadratic_roots", "cancellation_bits_lost"]

# dtype -> (format name, same-width unsigned integer type, exponent bits, fraction bits)
_FORMATS = {
    np.dtype(np.float16): ("binary16", np.uint16, 5, 10),
    np.dtype(np.float32): ("binary32", np.uint32, 8, 23),
    np.dtype(np.float64): ("binary64", np.uint64, 11, 52),
}


def _format(dtype) -> tuple:
    key = np.dtype(dtype)
    if key not in _FORMATS:
        raise ValueError(f"dtype must be float16, float32 or float64, got {key}")
    return (key.type,) + _FORMATS[key]


def machine_epsilon(dtype=np.float64) -> float:
    r"""Machine epsilon: the gap between 1 and the next larger float.

    Found the classic way: halve a candidate :math:`\varepsilon` while
    :math:`1 + \varepsilon/2` still rounds to something larger than 1,
    with every operation carried out in `dtype`. For a :math:`t`-bit
    significand the answer is :math:`2^{1-t}`; the unit roundoff, the
    largest relative error of a single rounding, is half of it.

    Parameters
    ----------
    dtype : {numpy.float16, numpy.float32, numpy.float64}

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> machine_epsilon() == 2.0**-52
    True
    >>> machine_epsilon(np.float32) == 2.0**-23
    True
    """
    ftype = _format(dtype)[0]
    one, two = ftype(1), ftype(2)
    eps = ftype(1)
    while one + eps / two > one:
        eps = eps / two
    return float(eps)


def float_bits(x: float, dtype=np.float64) -> FloatBits:
    r"""Decode the IEEE 754 bit fields of ``x`` rounded to `dtype`.

    A binary interchange format stores a sign bit :math:`s`, a biased
    exponent field :math:`E`, and a fraction field :math:`F` of :math:`f`
    bits. With bias :math:`b = 2^{k-1} - 1` for a :math:`k`-bit exponent:

    - :math:`0 < E < 2^k - 1` (normal): :math:`(-1)^s (1 + F/2^f)\,2^{E - b}`;
    - :math:`E = 0` (zero or subnormal): :math:`(-1)^s (F/2^f)\,2^{1 - b}`,
      filling the gap below the smallest normal number evenly;
    - :math:`E = 2^k - 1`: infinity if :math:`F = 0`, NaN otherwise.

    Parameters
    ----------
    x : float
    dtype : {numpy.float16, numpy.float32, numpy.float64}

    Returns
    -------
    FloatBits

    Examples
    --------
    >>> bits = float_bits(0.1)
    >>> bits.sign, bits.exponent, bits.exponent_bits
    (0, -4, '01111111011')
    >>> bits.fraction_bits[:12]  # 0.1 = 1.6 * 2**-4, and 0.6 is 0.1001 1001 ... in binary
    '100110011001'
    >>> bits.significand * 2.0**bits.exponent == 0.1
    True
    >>> float_bits(5e-324).category  # the smallest positive binary64 number
    'subnormal'
    """
    ftype, name, utype, e_bits, f_bits = _format(dtype)
    stored = np.array(x, dtype=ftype)
    total = 1 + e_bits + f_bits
    word = format(int(stored.view(utype)), f"0{total}b")
    sign = int(word[0])
    exponent_field, fraction_field = word[1 : 1 + e_bits], word[1 + e_bits :]
    E, F = int(exponent_field, 2), int(fraction_field, 2)
    bias = 2 ** (e_bits - 1) - 1
    if E == 2**e_bits - 1:
        category = "nan" if F else "infinity"
        exponent, significand = E - bias, math.nan
    elif E == 0:
        category = "subnormal" if F else "zero"
        exponent, significand = 1 - bias, F / 2**f_bits
    else:
        category = "normal"
        exponent, significand = E - bias, 1.0 + F / 2**f_bits
    return FloatBits(
        value=float(stored),
        format=name,
        sign=sign,
        exponent_bits=exponent_field,
        fraction_bits=fraction_field,
        exponent=exponent,
        significand=significand,
        category=category,
    )


def ulp(x, dtype=np.float64):
    r"""Unit in the last place: the gap from :math:`|x|` to the next larger float.

    It doubles at every power of two, so floating-point numbers have
    constant *relative* spacing: :math:`\mathrm{ulp}(x) \approx \varepsilon |x|`.
    Thin wrapper around :func:`numpy.spacing`.

    Parameters
    ----------
    x : float or array-like of float
    dtype : {numpy.float16, numpy.float32, numpy.float64}

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> ulp(1.0) == machine_epsilon()
    True
    >>> ulp(2.0**60)  # beyond 2**53 not every integer is representable
    256.0
    """
    ftype = _format(dtype)[0]
    out = np.spacing(np.abs(np.asarray(x, dtype=ftype)))
    return float(out) if out.ndim == 0 else out


def toy_float_system(precision: int = 3, emin: int = -1, emax: int = 3, subnormals: bool = False) -> np.ndarray:
    r"""Every positive number of a small binary floating-point system.

    The numbers are :math:`m \cdot 2^{e - t}` with :math:`t` = `precision`,
    :math:`e_{\min} \le e \le e_{\max}`, and normalized significands
    :math:`2^{t-1} \le m \le 2^t - 1`, following Higham, *Accuracy and
    Stability of Numerical Algorithms*, 2nd ed., sec. 2.1 (whose Figure
    2.1 is the default system). Within each binade :math:`[2^{e-1}, 2^e)`
    the numbers are equally spaced, and the spacing doubles from one
    binade to the next. With `subnormals`, the numbers
    :math:`m \cdot 2^{e_{\min} - t}` for :math:`0 < m < 2^{t-1}` are
    included too, filling the gap between 0 and the smallest normal
    number.

    Parameters
    ----------
    precision : int
        Significand bits :math:`t`, including the leading bit.
    emin, emax : int
        Exponent range.
    subnormals : bool

    Returns
    -------
    ndarray, sorted ascending

    Examples
    --------
    >>> toy_float_system(precision=3, emin=0, emax=1).tolist()
    [0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75]
    >>> toy_float_system(precision=3, emin=0, emax=0, subnormals=True).tolist()
    [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    """
    if precision < 1 or emax < emin:
        raise ValueError("need precision >= 1 and emax >= emin")
    m = np.arange(2 ** (precision - 1), 2**precision, dtype=np.float64)
    values = [m * 2.0 ** (e - precision) for e in range(emin, emax + 1)]
    if subnormals:
        values.append(np.arange(1, 2 ** (precision - 1), dtype=np.float64) * 2.0 ** (emin - precision))
    return np.sort(np.concatenate(values))


def quadratic_roots(a: float, b: float, c: float, stable: bool = True) -> np.ndarray:
    r"""Real roots of :math:`ax^2 + bx + c`, with or without cancellation.

    The textbook formula :math:`(-b \pm \sqrt{b^2 - 4ac})/(2a)` subtracts
    nearly equal numbers for the smaller-magnitude root whenever
    :math:`b^2 \gg |4ac|`, so that root loses most of its digits. The
    stable version (``stable=True``) computes only the root without
    cancellation,
    :math:`q = -\tfrac12\bigl(b + \operatorname{sign}(b)\sqrt{b^2 - 4ac}\bigr)`,
    :math:`x_1 = q/a`, and gets the other from the product of the roots,
    :math:`x_2 = c/q` (Forsythe 1966; Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., sec. 1.8).

    Parameters
    ----------
    a, b, c : float
        Coefficients, ``a != 0``, with nonnegative discriminant.
    stable : bool
        Use the cancellation-free formulas; ``False`` gives the textbook formula.

    Returns
    -------
    ndarray, shape (2,)
        Both roots, in ascending order.

    Examples
    --------
    >>> naive = quadratic_roots(1.0, 1e8, 1.0, stable=False)
    >>> stable = quadratic_roots(1.0, 1e8, 1.0)
    >>> float(stable[1])  # the small root is -1e-8 to 16 digits
    -1e-08
    >>> bool(abs(naive[1] + 1e-8) / 1e-8 > 0.2)  # the textbook formula is off by over 20%
    True
    """
    if a == 0.0:
        raise ValueError("a must be nonzero")
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError("complex roots: the discriminant b^2 - 4ac is negative")
    sqrt_disc = math.sqrt(disc)
    if not stable:
        roots = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]
    else:
        q = -0.5 * (b + math.copysign(sqrt_disc, b))
        roots = [q / a, c / q] if q != 0.0 else [0.0, 0.0]
    return np.sort(np.array(roots))


def cancellation_bits_lost(x: float, y: float) -> float:
    r"""Significant bits lost when computing :math:`x - y`.

    The loss-of-precision theorem: if :math:`2^{-q} \le |1 - y/x| \le 2^{-p}`,
    then at most :math:`q` and at least :math:`p` significant binary bits
    are lost in the subtraction :math:`x - y`. This returns the continuous
    estimate :math:`-\log_2 |1 - y/x|`, clipped at zero (no loss when
    :math:`y/x \le 0` or :math:`y/x \ge 2`). See W. Cheney and D. Kincaid, *Numerical
    Mathematics and Computing*, 7th ed. (Brooks/Cole, 2013), sec. 2.3.

    Parameters
    ----------
    x, y : float
        The operands, ``x != 0``.

    Returns
    -------
    float
        Bits lost; ``inf`` when ``x == y``.

    Examples
    --------
    >>> cancellation_bits_lost(1.0, 0.75)
    2.0
    >>> round(cancellation_bits_lost(1.0, 1.0 - 1e-12), 1)  # about 40 of binary64's 53 bits
    39.9
    """
    if x == 0.0:
        raise ValueError("x must be nonzero")
    rel = abs(1.0 - y / x)
    if rel == 0.0:
        return math.inf
    return max(0.0, -math.log2(rel))
