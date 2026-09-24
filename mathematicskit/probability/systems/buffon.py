r"""Buffon's needle: a geometric probability that measures :math:`\pi`.

A needle of length :math:`\ell` dropped at random on a floor ruled with
parallel lines a distance :math:`d \geq \ell` apart crosses a line with
probability :math:`2\ell/(\pi d)` (Buffon, 1777). No numpy/scipy routine
simulates this, so the drop is hand-written on top of
:func:`numpy.random.default_rng`. See Ross, *A First Course in
Probability*, 9th ed., Example 6.3b.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.probability.core.base import BuffonNeedleResult

__all__ = ["buffon_needle"]


def buffon_needle(n_drops: int = 100000, length: float = 1.0, spacing: float = 1.0, seed: int = 0) -> BuffonNeedleResult:
    r"""Drop ``n_drops`` needles on a ruled floor and estimate :math:`\pi` from the crossings.

    Each drop places the needle's center a uniform distance
    :math:`y \in [0, d/2]` from the nearest line and points it in a
    uniformly random direction. The needle crosses that line when
    :math:`y \leq \tfrac{\ell}{2}\sin\theta`, which happens with probability

    .. math::

       P(\text{cross}) = \frac{2\ell}{\pi d}, \qquad \ell \leq d,

    so :math:`\hat\pi = 2\ell n / (d \cdot \#\text{crossings})`. The random
    direction comes from normalizing a standard 2D Gaussian vector, so the
    simulation never uses :math:`\pi` itself. See G.-L. Leclerc de Buffon,
    *Essai d'arithmétique morale* (1777), Sec. 23.

    Parameters
    ----------
    n_drops : int
        Number of needles dropped.
    length : float
        Needle length :math:`\ell`.
    spacing : float
        Distance :math:`d` between the lines; must satisfy ``length <= spacing``.
    seed : int

    Returns
    -------
    BuffonNeedleResult

    Examples
    --------
    >>> result = buffon_needle(n_drops=400000, seed=0)
    >>> bool(abs(result.pi_estimate - 3.14159) < 0.03)
    True
    >>> round(result.exact_crossing_probability, 4)
    0.6366
    """
    if not 0.0 < length <= spacing:
        raise ValueError("buffon_needle requires 0 < length <= spacing (the short-needle case).")
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, spacing / 2.0, size=n_drops)
    direction = rng.standard_normal(size=(n_drops, 2))
    sin_theta = np.abs(direction[:, 1]) / np.linalg.norm(direction, axis=1)
    crossings = int(np.count_nonzero(y <= 0.5 * length * sin_theta))
    fraction = crossings / n_drops
    pi_estimate = float(2.0 * length * n_drops / (spacing * crossings)) if crossings else float("inf")
    return BuffonNeedleResult(
        pi_estimate=pi_estimate,
        crossing_fraction=float(fraction),
        exact_crossing_probability=float(2.0 * length / (np.pi * spacing)),
        n_drops=int(n_drops),
    )
