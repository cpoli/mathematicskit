r"""The Lambert W function, via :mod:`scipy.special`.

:math:`W(z)` is the inverse of :math:`w \mapsto we^w`. Johann Heinrich
Lambert's 1758 series for the trinomial equation and Euler's 1783
follow-up contain it implicitly; the modern name, notation, and branch
conventions come from R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J.
Jeffrey, and D. E. Knuth, "On the Lambert W function," Advances in
Computational Mathematics 5 (1996), 329-359.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["lambert_w"]


def lambert_w(z, branch=0):
    r"""The Lambert W function: the solution :math:`w` of :math:`we^w = z`.

    Via :func:`scipy.special.lambertw`. For real
    :math:`z \ge -1/e` the principal branch (``branch=0``) is real, and
    for :math:`-1/e \le z < 0` the lower branch ``branch=-1`` is a second
    real solution; in those cases the imaginary part (identically zero)
    is dropped and a real result is returned. See Corless et al. (1996).

    Parameters
    ----------
    z : float, complex, or array-like
    branch : int, optional
        Branch index ``k``; 0 is the principal branch.

    Returns
    -------
    float, complex, or ndarray

    Examples
    --------
    >>> import math
    >>> round(float(lambert_w(math.e)), 12)  # 1 * e^1 = e
    1.0
    >>> # both real branches at z = -ln(2)/2: W_0 = -ln 2, W_{-1} = -ln 4
    >>> round(float(lambert_w(-math.log(2) / 2)), 12) == round(-math.log(2), 12)
    True
    >>> round(float(lambert_w(-math.log(2) / 2, branch=-1)), 12) == round(-math.log(4), 12)
    True
    """
    w = special.lambertw(z, k=branch)
    z_arr = np.asarray(z)
    real_domain = not np.iscomplexobj(z_arr) and np.all(z_arr >= -1.0 / np.e)
    if branch == -1:
        real_domain = real_domain and np.all(z_arr < 0)
    if real_domain and branch in (0, -1):
        return np.real(w)
    return w
