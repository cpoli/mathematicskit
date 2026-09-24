"""Stationary iterative solvers: Jacobi, Gauss-Seidel, and successive over-relaxation (SOR).

Hand-rolled: scipy has no stationary-iteration solver, and the sweep
itself -- each unknown updated from its own equation using the latest
available values of the others -- is the pedagogical subject. Each
solver splits :math:`A = D + L + U` (diagonal, strictly lower, strictly
upper) and iterates :math:`x_{k+1} = G x_k + c`; it converges for every
starting vector if and only if the spectral radius :math:`\\rho(G) < 1`.
For consistently ordered matrices (e.g. the tridiagonal 1-D Poisson
matrix), Young's theory gives :math:`\\rho(G_{GS}) = \\rho(G_J)^2` and
the optimal relaxation factor of :func:`optimal_sor_omega`. See Saad,
*Iterative Methods for Sparse Linear Systems*, 2nd ed., 2003, Ch. 4, and
D. M. Young, *Iterative Solution of Large Linear Systems*, 1971.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathematicskit.linalg.core.base import IterativeLinearSolver, IterativeSolveResult

__all__ = ["JacobiIteration", "GaussSeidel", "SOR", "jacobi_spectral_radius", "optimal_sor_omega"]


def _prepare(a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray]):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    if a.shape != (n, n):
        raise ValueError("a must be square with shape (n, n) matching b")
    if np.any(np.diag(a) == 0.0):
        raise ValueError("a must have a nonzero diagonal")
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64)
    return a, b, x


class SOR(IterativeLinearSolver):
    r"""Successive over-relaxation (Frankel 1950, Young 1950).

    One sweep updates the unknowns in order, blending the Gauss-Seidel
    value with the old one:

    .. math::

       x_i \leftarrow (1 - \omega)\, x_i + \frac{\omega}{a_{ii}}
       \Big(b_i - \sum_{j < i} a_{ij} x_j^{\text{new}} - \sum_{j > i} a_{ij} x_j^{\text{old}}\Big).

    ``omega = 1`` is Gauss-Seidel; ``1 < omega < 2`` over-relaxes. For
    symmetric positive-definite ``A`` it converges for every
    ``0 < omega < 2`` (Ostrowski-Reich).

    Parameters
    ----------
    omega : float
        Relaxation factor, ``0 < omega < 2``.
    tol : float
        Relative residual tolerance, ``||b - A x_k|| < tol * ||b||``.
    max_iter : int
        Maximum number of sweeps.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([1.0, 2.0])
    >>> result = SOR(omega=1.1, tol=1e-12).solve(A, b)
    >>> np.allclose(A @ result.x, b), result.method
    (True, 'sor')
    """

    method = "sor"

    def __init__(self, omega: float = 1.0, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        if not 0.0 < omega < 2.0:
            raise ValueError("omega must lie in (0, 2)")
        self.omega = float(omega)

    def _sweep(self, a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        w = self.omega
        for i in range(b.shape[0]):
            sigma = a[i, :i] @ x[:i] + a[i, i + 1 :] @ x[i + 1 :]
            x[i] = (1.0 - w) * x[i] + w * (b[i] - sigma) / a[i, i]
        return x

    def solve(self, a: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> IterativeSolveResult:
        a, b, x = _prepare(a, b, x0)
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-300:
            return IterativeSolveResult(x=np.zeros_like(b), residual_history=np.array([0.0]), iterations=0, converged=True, method=self.method)
        residuals = [float(np.linalg.norm(b - a @ x) / b_norm)]
        converged = residuals[0] < self.tol
        it = 0
        while not converged and it < self.max_iter:
            it += 1
            x = self._sweep(a, b, x)
            residuals.append(float(np.linalg.norm(b - a @ x) / b_norm))
            if not np.isfinite(residuals[-1]):
                break
            converged = residuals[-1] < self.tol
        return IterativeSolveResult(x=x, residual_history=np.array(residuals), iterations=it, converged=converged, method=self.method)


class GaussSeidel(SOR):
    r"""Gauss-Seidel iteration (Gauss 1823, Seidel 1874): :class:`SOR` with ``omega = 1``.

    Each unknown is solved from its own equation using the *newest*
    values of the unknowns already updated in this sweep. Converges for
    strictly diagonally dominant or symmetric positive-definite ``A``.

    Parameters
    ----------
    tol : float
    max_iter : int

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([1.0, 2.0])
    >>> result = GaussSeidel(tol=1e-12).solve(A, b)
    >>> np.round(result.x, 8)
    array([0.09090909, 0.63636364])
    """

    method = "gauss_seidel"

    def __init__(self, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(omega=1.0, tol=tol, max_iter=max_iter)


class JacobiIteration(SOR):
    r"""Jacobi iteration (Jacobi 1845): every unknown updated *simultaneously* from the previous iterate.

    :math:`x_{k+1} = D^{-1}(b - (L + U) x_k)`. Simpler and fully parallel,
    but for consistently ordered matrices it needs about twice as many
    sweeps as Gauss-Seidel.

    Parameters
    ----------
    tol : float
    max_iter : int

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([1.0, 2.0])
    >>> result = JacobiIteration(tol=1e-12).solve(A, b)
    >>> np.allclose(A @ result.x, b)
    True
    """

    method = "jacobi"

    def __init__(self, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(omega=1.0, tol=tol, max_iter=max_iter)

    def _sweep(self, a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        d = np.diag(a)
        return (b - (a @ x - d * x)) / d


def jacobi_spectral_radius(a: np.ndarray) -> float:
    r"""Spectral radius :math:`\rho(G_J)` of the Jacobi iteration matrix :math:`G_J = I - D^{-1} A`.

    Computed with :func:`numpy.linalg.eigvals`. The Jacobi iteration
    converges from every starting vector iff this is below 1.

    Parameters
    ----------
    a : ndarray, shape (n, n)

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> n = 5
    >>> A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    >>> bool(np.isclose(jacobi_spectral_radius(A), np.cos(np.pi / (n + 1))))
    True
    """
    a = np.asarray(a, dtype=np.float64)
    d = np.diag(a)
    if np.any(d == 0.0):
        raise ValueError("a must have a nonzero diagonal")
    g = np.eye(a.shape[0]) - a / d[:, None]
    return float(np.max(np.abs(np.linalg.eigvals(g))))


def optimal_sor_omega(a: np.ndarray) -> float:
    r"""Young's optimal SOR relaxation factor :math:`\omega^* = 2 / (1 + \sqrt{1 - \rho(G_J)^2})`.

    Exact for consistently ordered matrices whose Jacobi iteration matrix
    has real eigenvalues with :math:`\rho(G_J) < 1` (e.g. the discrete
    Poisson matrices); at :math:`\omega^*` the SOR spectral radius drops
    to :math:`\omega^* - 1`. See D. M. Young, Trans. AMS 76 (1954),
    92-111.

    Parameters
    ----------
    a : ndarray, shape (n, n)

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> A = 2 * np.eye(3) - np.eye(3, k=1) - np.eye(3, k=-1)  # rho_J = cos(pi/4)
    >>> round(optimal_sor_omega(A), 6)  # 2 / (1 + sqrt(1 - cos(pi/4)^2))
    1.171573
    """
    rho = jacobi_spectral_radius(a)
    if rho >= 1.0:
        raise ValueError("Jacobi iteration does not converge (rho >= 1); Young's formula does not apply")
    return float(2.0 / (1.0 + np.sqrt(1.0 - rho**2)))
