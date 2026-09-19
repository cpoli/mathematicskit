r"""Mandelbrot and Julia set generation via escape-time iteration.

Both iterate :math:`z_{n+1} = z_n^2 + c` and record how many iterations
it takes :math:`|z_n|` to exceed 2 (beyond which the orbit provably
diverges to infinity), capping at `max_iter` for points that appear
bounded. The per-pixel inner loop is the classic Numba
factory-function pattern documented in physicskit's ``core/`` modules: a
module-level factory closes over nothing here (the grid itself is the
input) and returns a standalone ``@njit`` kernel. See Devaney, *A First
Course in Chaotic Dynamical Systems*, 2nd ed., Ch. 15 ("The Mandelbrot
Set"), and Peitgen & Richter, *The Beauty of Fractals*, 1986.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mathematicskit.fractals_chaos.core.base import EscapeTimeResult

__all__ = ["mandelbrot_set", "julia_set"]


@njit(cache=True)
def _escape_time_mandelbrot(c_re, c_im, max_iter):
    ny, nx = c_re.shape
    out = np.empty((ny, nx), dtype=np.int64)
    for i in range(ny):
        for j in range(nx):
            zre, zim = 0.0, 0.0
            cre, cim = c_re[i, j], c_im[i, j]
            n = 0
            while n < max_iter and zre * zre + zim * zim <= 4.0:
                zre, zim = zre * zre - zim * zim + cre, 2.0 * zre * zim + cim
                n += 1
            out[i, j] = n
    return out


@njit(cache=True)
def _escape_time_julia(z_re, z_im, c_re, c_im, max_iter):
    ny, nx = z_re.shape
    out = np.empty((ny, nx), dtype=np.int64)
    for i in range(ny):
        for j in range(nx):
            zre, zim = z_re[i, j], z_im[i, j]
            n = 0
            while n < max_iter and zre * zre + zim * zim <= 4.0:
                zre, zim = zre * zre - zim * zim + c_re, 2.0 * zre * zim + c_im
                n += 1
            out[i, j] = n
    return out


def mandelbrot_set(extent=(-2.0, 1.0, -1.5, 1.5), resolution: int = 400, max_iter: int = 100) -> EscapeTimeResult:
    r"""Escape-time grid for the Mandelbrot set: :math:`z_0 = 0`, ``c`` sweeps the grid.

    The Mandelbrot set is :math:`\{c \in \mathbb{C} : z_{n+1}=z_n^2+c,\
    z_0=0 \text{ stays bounded}\}`. See Devaney, *A First Course in
    Chaotic Dynamical Systems*, 2nd ed., Ch. 15.

    Parameters
    ----------
    extent : tuple of float
        ``(re_min, re_max, im_min, im_max)`` region of the complex plane.
    resolution : int
        Number of pixels along the real axis (the imaginary-axis count is
        scaled to keep pixels roughly square).
    max_iter : int
        Iteration cap.

    Returns
    -------
    EscapeTimeResult

    Examples
    --------
    >>> result = mandelbrot_set(resolution=10, max_iter=50)
    >>> result.iterations.shape[1]
    10
    >>> # c = -1 is in the Mandelbrot set (a period-2 cycle -1, 0, -1, ...).
    >>> from mathematicskit.fractals_chaos.systems.mandelbrot_julia import _escape_time_mandelbrot
    >>> import numpy as np
    >>> int(_escape_time_mandelbrot(np.array([[-1.0]]), np.array([[0.0]]), 200)[0, 0])
    200
    """
    re_min, re_max, im_min, im_max = extent
    nx = int(resolution)
    ny = max(int(round(nx * (im_max - im_min) / (re_max - re_min))), 1)
    re = np.linspace(re_min, re_max, nx)
    im = np.linspace(im_min, im_max, ny)
    c_re, c_im = np.meshgrid(re, im)
    iterations = _escape_time_mandelbrot(c_re, c_im, int(max_iter))
    return EscapeTimeResult(iterations=iterations, extent=tuple(extent), max_iter=int(max_iter))


def julia_set(c: complex, extent=(-2.0, 2.0, -2.0, 2.0), resolution: int = 400, max_iter: int = 100) -> EscapeTimeResult:
    r"""Escape-time grid for the filled Julia set of :math:`z \mapsto z^2 + c`, ``c`` fixed.

    Unlike the Mandelbrot set (which sweeps ``c`` with a fixed
    :math:`z_0=0`), the Julia set fixes ``c`` and sweeps the *initial
    condition* :math:`z_0` over the grid. See Devaney, *A First Course in
    Chaotic Dynamical Systems*, 2nd ed., Ch. 15.

    Parameters
    ----------
    c : complex
        The fixed parameter.
    extent : tuple of float
        ``(re_min, re_max, im_min, im_max)`` region of the complex plane.
    resolution : int
        Number of pixels along the real axis.
    max_iter : int
        Iteration cap.

    Returns
    -------
    EscapeTimeResult

    Examples
    --------
    >>> result = julia_set(c=-0.4 + 0.6j, resolution=10, max_iter=50)
    >>> result.iterations.shape[1]
    10
    >>> # For c=0, z_{n+1}=z_n^2: |z0|<1 always stays bounded.
    >>> from mathematicskit.fractals_chaos.systems.mandelbrot_julia import _escape_time_julia
    >>> import numpy as np
    >>> int(_escape_time_julia(np.array([[0.5]]), np.array([[0.0]]), 0.0, 0.0, 200)[0, 0])
    200
    """
    re_min, re_max, im_min, im_max = extent
    nx = int(resolution)
    ny = max(int(round(nx * (im_max - im_min) / (re_max - re_min))), 1)
    re = np.linspace(re_min, re_max, nx)
    im = np.linspace(im_min, im_max, ny)
    z_re, z_im = np.meshgrid(re, im)
    iterations = _escape_time_julia(z_re, z_im, float(np.real(c)), float(np.imag(c)), int(max_iter))
    return EscapeTimeResult(iterations=iterations, extent=tuple(extent), max_iter=int(max_iter))
