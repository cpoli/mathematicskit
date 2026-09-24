r"""Gauss's hypergeometric function and Kummer's confluent hypergeometric function, via :mod:`scipy.special`.

:math:`{}_2F_1` (:func:`scipy.special.hyp2f1`) and :math:`{}_1F_1`
(:func:`scipy.special.hyp1f1`) specialize to most of the classical
special functions -- logarithms, inverse trigonometric functions,
orthogonal polynomials, incomplete beta/gamma functions, Bessel
functions. See NIST *Digital Library of Mathematical Functions*,
Ch. 13 and 15.
"""

from __future__ import annotations

from scipy import special

__all__ = ["hypergeometric_2f1", "confluent_hypergeometric_1f1"]


def hypergeometric_2f1(a, b, c, z):
    r"""Gauss's hypergeometric function :math:`{}_2F_1(a, b; c; z) = \sum_{n=0}^\infty \frac{(a)_n (b)_n}{(c)_n}\frac{z^n}{n!}`.

    :math:`(q)_n = q(q+1)\cdots(q+n-1)` is the rising factorial. The
    series converges for :math:`|z| < 1`; :func:`scipy.special.hyp2f1`
    uses analytic continuation elsewhere. Gauss's summation theorem
    (1812) evaluates it at :math:`z = 1` for :math:`\operatorname{Re}(c - a - b) > 0`:

    .. math::

       {}_2F_1(a, b; c; 1) = \frac{\Gamma(c)\Gamma(c-a-b)}{\Gamma(c-a)\Gamma(c-b)}.

    See C. F. Gauss, "Disquisitiones generales circa seriem infinitam,"
    Commentationes Societatis Regiae Scientiarum Gottingensis Recentiores
    2 (1813).

    Parameters
    ----------
    a, b, c : float
    z : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> # ln(1 + z) = z * 2F1(1, 1; 2; -z)
    >>> round(0.5 * float(hypergeometric_2f1(1.0, 1.0, 2.0, -0.5)), 12) == round(math.log(1.5), 12)
    True
    """
    return special.hyp2f1(a, b, c, z)


def confluent_hypergeometric_1f1(a, b, z):
    r"""Kummer's confluent hypergeometric function :math:`M(a, b, z) = {}_1F_1(a; b; z) = \sum_{n=0}^\infty \frac{(a)_n}{(b)_n}\frac{z^n}{n!}`.

    It solves Kummer's equation :math:`zw'' + (b - z)w' - aw = 0` and
    arises as the confluent limit
    :math:`{}_1F_1(a; b; z) = \lim_{c\to\infty} {}_2F_1(a, c; b; z/c)`.
    Kummer's transformation reads
    :math:`M(a, b, z) = e^z M(b - a, b, -z)`. Via
    :func:`scipy.special.hyp1f1`. See E. E. Kummer, "De integralibus
    quibusdam definitis et seriebus infinitis," Journal für die reine
    und angewandte Mathematik 17 (1837), 228-242.

    Parameters
    ----------
    a, b : float
    z : float or array-like of float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> round(float(confluent_hypergeometric_1f1(2.0, 2.0, 1.0)), 12) == round(math.e, 12)  # 1F1(a; a; z) = e^z
    True
    """
    return special.hyp1f1(a, b, z)
