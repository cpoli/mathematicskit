r"""Diffusion-limited aggregation (DLA): a cluster grown from random walkers
that stick on first contact.

The random-walk inner loop is compiled with Numba, following the
fractals_chaos convention for performance-critical kernels. See T. A.
Witten and L. M. Sander, "Diffusion-Limited Aggregation, a Kinetic
Critical Phenomenon," Physical Review Letters 47(19) (1981), 1400-1403.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["dla_cluster"]


@njit(cache=True)
def _grow(n_particles, size, seed):
    np.random.seed(seed)
    grid = np.zeros((size, size), dtype=np.bool_)
    c = size // 2
    grid[c, c] = True
    coords = np.zeros((n_particles, 2), dtype=np.int64)
    coords[0, 0], coords[0, 1] = 0, 0
    r_max = 1.0
    count = 1
    dx = np.array([1, -1, 0, 0])
    dy = np.array([0, 0, 1, -1])
    while count < n_particles:
        r_launch = r_max + 5.0
        r_kill = 2.0 * r_launch + 10.0
        if r_kill >= c - 2:
            break
        phi = 2.0 * np.pi * np.random.random()
        x = c + int(r_launch * np.cos(phi))
        y = c + int(r_launch * np.sin(phi))
        while True:
            k = np.random.randint(4)
            x += dx[k]
            y += dy[k]
            r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
            if r > r_kill:
                phi = 2.0 * np.pi * np.random.random()
                x = c + int(r_launch * np.cos(phi))
                y = c + int(r_launch * np.sin(phi))
                continue
            if grid[x + 1, y] or grid[x - 1, y] or grid[x, y + 1] or grid[x, y - 1]:
                grid[x, y] = True
                coords[count, 0], coords[count, 1] = x - c, y - c
                count += 1
                if r > r_max:
                    r_max = r
                break
    return coords[:count]


def dla_cluster(n_particles: int, seed: int = 0, size: int = 801) -> np.ndarray:
    r"""Grow a two-dimensional DLA cluster on the square lattice.

    Particles are released one at a time on a circle just outside the
    cluster and random-walk until they touch it, then stick. Walkers
    that stray too far are relaunched. The branched clusters that result
    have fractal dimension about 1.71.

    Parameters
    ----------
    n_particles : int
        Target cluster size, including the seed particle.
    seed : int
        Random seed.
    size : int
        Lattice width; growth stops early if the cluster nears the edge.

    Returns
    -------
    ndarray, shape (m, 2), int
        Lattice coordinates of the cluster's particles relative to the
        seed, in the order they stuck (``m <= n_particles``).

    Examples
    --------
    >>> cluster = dla_cluster(200, seed=1)
    >>> cluster.shape
    (200, 2)
    >>> cluster[0].tolist()
    [0, 0]
    """
    return _grow(int(n_particles), int(size), int(seed))
