r"""Piecewise-linear (P1) Galerkin finite elements for the 1D Poisson problem.

Instead of differencing the equation, the finite element method asks
the weak form :math:`-\int u' v' = \int f v` to hold for every
piecewise-linear "hat" function :math:`v = \phi_i` on a mesh (Courant,
1943; Turner, Clough, Martin & Topp, 1956). With :math:`u = \sum_j u_j \phi_j`
this is the sparse system :math:`K U = -F`, with stiffness
:math:`K_{ij} = \int \phi_i' \phi_j'` and load :math:`F_i = \int f \phi_i`.
The mesh may be nonuniform, and in 1D the nodal values are exact whenever
the load integrals are (the discrete Green's function is exact). The load
is integrated per element with Gauss-Legendre quadrature
(:func:`numpy.polynomial.legendre.leggauss`); the system is solved with
:func:`scipy.sparse.linalg.spsolve`. See Strang & Fix, *An Analysis of
the Finite Element Method*, 2nd ed., 2008, Ch. 1, and Larson & Bengzon,
*The Finite Element Method*, 2013, Ch. 2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from mathematicskit.pde.core.base import EllipticSolution

__all__ = ["fem_poisson_1d"]


def fem_poisson_1d(
    f: float | Callable,
    a: float = 0.0,
    b: float = 1.0,
    n_elements: int = 10,
    ua: float = 0.0,
    ub: float = 0.0,
    nodes: Optional[np.ndarray] = None,
    quad_points: int = 3,
) -> EllipticSolution:
    r"""Solve :math:`u'' = f` on :math:`[a, b]`, :math:`u(a) = u_a`, :math:`u(b) = u_b`, with P1 finite elements.

    Parameters
    ----------
    f : float or callable
        Source term, vectorized over NumPy arrays if callable.
    a, b : float
        Interval, ignored when `nodes` is given.
    n_elements : int
        Number of equal elements, ignored when `nodes` is given.
    ua, ub : float
        Boundary values.
    nodes : ndarray, optional
        Strictly increasing mesh nodes (a nonuniform mesh), endpoints included.
    quad_points : int
        Gauss-Legendre points per element for the load integrals (exact
        for ``f`` polynomial of degree ``<= 2 * quad_points - 2``).

    Returns
    -------
    EllipticSolution
        ``x`` are the mesh nodes and ``u`` the nodal values.

    Examples
    --------
    >>> import numpy as np
    >>> nodes = np.array([0.0, 0.1, 0.35, 0.4, 0.8, 1.0])  # deliberately uneven
    >>> sol = fem_poisson_1d(lambda x: 6 * x, nodes=nodes)  # exact u = x^3 - x
    >>> bool(np.allclose(sol.u, nodes**3 - nodes))  # exact at the nodes
    True
    """
    x = np.linspace(a, b, n_elements + 1) if nodes is None else np.asarray(nodes, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("nodes must be strictly increasing with at least two elements")
    h = np.diff(x)
    n = len(x)

    # Stiffness: each element contributes (1/h) [[1, -1], [-1, 1]].
    main = np.zeros(n)
    main[:-1] += 1.0 / h
    main[1:] += 1.0 / h
    K = sp.diags([-1.0 / h, main, -1.0 / h], [-1, 0, 1], format="csr")

    # Load: F_i = sum over elements of int f phi_i, by Gauss-Legendre on each element.
    s, w = np.polynomial.legendre.leggauss(quad_points)
    xq = 0.5 * (x[:-1, None] + x[1:, None]) + 0.5 * h[:, None] * s[None, :]
    fq = np.asarray(f(xq), dtype=np.float64) * np.ones(xq.shape) if callable(f) else np.full(xq.shape, float(f))
    left_hat = 0.5 * (1.0 - s)  # phi of the element's left node at the Gauss points
    weights = 0.5 * h[:, None] * w[None, :] * fq
    F = np.zeros(n)
    F[:-1] += weights @ left_hat
    F[1:] += weights @ (1.0 - left_hat)

    interior = slice(1, n - 1)
    K_int = K[interior, interior]
    rhs = -F[interior] - K[interior, 0].toarray().ravel() * ua - K[interior, n - 1].toarray().ravel() * ub
    u_int = sla.spsolve(K_int.tocsc(), rhs)
    u = np.concatenate([[ua], np.atleast_1d(u_int), [ub]])
    return EllipticSolution(x=x, u=u, method="fem", residual_norm=float(np.linalg.norm(K_int @ u_int - rhs)))
