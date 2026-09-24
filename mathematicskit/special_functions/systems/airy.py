r"""The Airy functions, via :mod:`scipy.special`.

:math:`\operatorname{Ai}` and :math:`\operatorname{Bi}` are the two
standard solutions of Airy's equation :math:`y'' = xy`, the simplest
second-order ODE whose solutions switch from oscillating (:math:`x<0`)
to exponential (:math:`x>0`) behaviour. See G. B. Airy, "On the
intensity of light in the neighbourhood of a caustic," Transactions of
the Cambridge Philosophical Society 6 (1838), 379-402, and NIST
*Digital Library of Mathematical Functions*, Ch. 9.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from mathematicskit.special_functions.core.base import AiryResult

__all__ = ["airy_functions"]


def airy_functions(x):
    r"""The Airy functions :math:`\operatorname{Ai}, \operatorname{Ai}', \operatorname{Bi}, \operatorname{Bi}'` at ``x``.

    :math:`\operatorname{Ai}(x) = \frac1\pi\int_0^\infty \cos(t^3/3 + xt)\,dt`
    decays as :math:`x\to+\infty`; :math:`\operatorname{Bi}` grows. Both
    satisfy :math:`y'' = xy`. Via :func:`scipy.special.airy`.

    Parameters
    ----------
    x : float or array-like of float

    Returns
    -------
    AiryResult

    Examples
    --------
    >>> from scipy.special import gamma
    >>> r = airy_functions(0.0)
    >>> round(float(r.ai), 12) == round(1 / (3 ** (2 / 3) * float(gamma(2 / 3))), 12)
    True
    """
    x = np.asarray(x, dtype=float)
    ai, aip, bi, bip = special.airy(x)
    return AiryResult(x=x, ai=ai, ai_prime=aip, bi=bi, bi_prime=bip)
