r"""Sparse finite-difference operators and uniform grids for mathematicskit.pde.

The second-difference matrices are assembled with :func:`scipy.sparse.diags`
and :func:`scipy.sparse.kron` rather than by hand: the 2D five-point
Laplacian on an ``"ij"``-ordered grid is the Kronecker sum
:math:`L_x \otimes I_y + I_x \otimes L_y`. See LeVeque, *Finite Difference
Methods for Ordinary and Partial Differential Equations*, SIAM 2007,
Ch. 2.4 and 3.3.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["uniform_grid", "laplacian_1d", "laplacian_2d"]


def uniform_grid(a: float, b: float, n: int, periodic: bool = False) -> tuple[np.ndarray, float]:
    """Return ``n`` equally spaced points on ``[a, b]`` and their spacing.

    Parameters
    ----------
    a, b : float
        Interval endpoints.
    n : int
        Number of points.
    periodic : bool
        If ``True``, omit the right endpoint ``b`` (it duplicates ``a``), so
        the spacing is ``(b - a) / n``; otherwise both endpoints are
        included and the spacing is ``(b - a) / (n - 1)``.

    Returns
    -------
    x : ndarray, shape (n,)
    dx : float

    Examples
    --------
    >>> x, dx = uniform_grid(0.0, 1.0, 5)
    >>> x.tolist(), dx
    ([0.0, 0.25, 0.5, 0.75, 1.0], 0.25)
    >>> uniform_grid(0.0, 1.0, 4, periodic=True)[0].tolist()
    [0.0, 0.25, 0.5, 0.75]
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    x = np.linspace(a, b, n, endpoint=not periodic)
    dx = (b - a) / (n if periodic else n - 1)
    return x, float(dx)


def laplacian_1d(n: int, dx: float, bc: str = "dirichlet") -> sp.csr_matrix:
    r"""Second-difference matrix approximating :math:`d^2/dx^2`, :math:`O(dx^2)`.

    .. math:: (L u)_i = \frac{u_{i-1} - 2 u_i + u_{i+1}}{dx^2}

    Parameters
    ----------
    n : int
        Number of unknowns.
    dx : float
        Grid spacing.
    bc : {"dirichlet", "periodic", "neumann"}
        ``"dirichlet"``: the `n` unknowns are interior points and the
        (known) boundary values are dropped -- add their contribution to
        the right-hand side separately. ``"periodic"``: ``u_{-1} = u_{n-1}``
        and ``u_n = u_0``. ``"neumann"``: the `n` unknowns include both
        endpoints, with zero slope imposed by reflecting ghost points
        (``u_{-1} = u_1``).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n, n)

    Examples
    --------
    >>> L = laplacian_1d(4, 1.0)
    >>> L.toarray().astype(int).tolist()
    [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    """
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    L = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    if bc == "periodic":
        L[0, n - 1] = 1.0
        L[n - 1, 0] = 1.0
    elif bc == "neumann":
        L[0, 1] = 2.0
        L[n - 1, n - 2] = 2.0
    elif bc != "dirichlet":
        raise ValueError(f"Unknown boundary condition '{bc}'")
    return (L / dx**2).tocsr()


def laplacian_2d(nx: int, ny: int, dx: float, dy: float) -> sp.csr_matrix:
    r"""Five-point Laplacian on an ``nx`` by ``ny`` grid of interior unknowns (Dirichlet).

    Unknown ``u[i, j]`` is stored at flat index ``i * ny + j`` (row-major,
    matching ``u.ravel()`` for an ``"ij"``-indexed array), so the operator
    is the Kronecker sum :math:`L_x \otimes I_{n_y} + I_{n_x} \otimes L_y`.

    Parameters
    ----------
    nx, ny : int
        Number of interior unknowns along x and y.
    dx, dy : float
        Grid spacings.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (nx * ny, nx * ny)

    Examples
    --------
    >>> import numpy as np
    >>> L = laplacian_2d(3, 3, 1.0, 1.0)
    >>> L.shape, float(L[4, 4])
    ((9, 9), -4.0)
    """
    Lx = laplacian_1d(nx, dx)
    Ly = laplacian_1d(ny, dy)
    return (sp.kron(Lx, sp.identity(ny)) + sp.kron(sp.identity(nx), Ly)).tocsr()
