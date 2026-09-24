r"""The Hausdorff distance between two point sets.

Built on :func:`scipy.spatial.distance.directed_hausdorff`, which
computes each one-sided distance efficiently. See F. Hausdorff,
*Grundzüge der Mengenlehre* (Leipzig: Veit, 1914), Ch. 8.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import directed_hausdorff

__all__ = ["hausdorff_distance"]


def hausdorff_distance(a, b) -> float:
    r"""The Hausdorff distance :math:`d_H(A, B) = \max\{\sup_{a}\inf_{b}\|a-b\|,\; \sup_{b}\inf_{a}\|a-b\|\}`.

    The smallest :math:`r` such that each set lies within distance
    :math:`r` of the other. Unlike the distance between closest points,
    it is zero only when the (closed) sets coincide.

    Parameters
    ----------
    a : array_like, shape (m, d)
    b : array_like, shape (n, d)

    Returns
    -------
    float

    Examples
    --------
    >>> hausdorff_distance([[0, 0], [1, 0]], [[0, 0], [1, 0], [1, 3]])
    3.0
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))
