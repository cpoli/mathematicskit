"""Shared "chaos game" driver for random iterated function systems --
supporting numerics for :mod:`mathematicskit.fractals_chaos.systems.ifs`, not a
model in its own right.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["chaos_game"]


def chaos_game(transforms: list[tuple[np.ndarray, np.ndarray, float]], n_points: int, seed: int = 0, x0: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Run the "chaos game": repeatedly apply a randomly chosen affine map.

    At each step, one of the ``(A_k, b_k, p_k)`` triples is chosen with
    probability :math:`p_k` and the current point :math:`x` is replaced
    by :math:`A_k x + b_k`. For a contractive *iterated function system*
    (every :math:`A_k` a contraction), the orbit converges (in
    distribution) onto the IFS's unique attractor regardless of the
    starting point -- the standard randomized algorithm for rendering
    self-similar fractals such as the Sierpinski triangle and the
    Barnsley fern. See Barnsley, *Fractals Everywhere*, 2nd ed., Ch. 3.

    Parameters
    ----------
    transforms : list of (ndarray, ndarray, float)
        ``(A, b, p)`` triples: a 2x2 matrix, a length-2 translation, and
        a selection probability (must sum to 1).
    n_points : int
        Number of points to generate.
    seed : int
        Random seed.
    x0 : ndarray, shape (2,), optional
        Starting point; defaults to the origin.

    Returns
    -------
    ndarray, shape (n_points, 2)

    Examples
    --------
    >>> import numpy as np
    >>> # A single constant map (A=0) with probability 1 sends every point to b.
    >>> transforms = [(np.zeros((2, 2)), np.array([1.0, 2.0]), 1.0)]
    >>> pts = chaos_game(transforms, n_points=5, x0=np.zeros(2))
    >>> np.allclose(pts, [[1.0, 2.0]] * 5)
    True
    """
    rng = np.random.default_rng(seed)
    probs = np.array([t[2] for t in transforms], dtype=np.float64)
    if not np.isclose(probs.sum(), 1.0, atol=1e-8):
        raise ValueError("transform probabilities must sum to 1")
    choices = rng.choice(len(transforms), size=n_points, p=probs)

    x = np.zeros(2) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    points = np.empty((n_points, 2))
    for i in range(n_points):
        a, b, _ = transforms[choices[i]]
        x = a @ x + b
        points[i] = x
    return points
