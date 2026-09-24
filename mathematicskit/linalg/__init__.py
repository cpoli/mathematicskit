"""mathematicskit.linalg: numerical linear algebra, built on numpy.linalg/scipy.linalg/scipy.sparse.linalg.

LU decomposition with partial pivoting (:func:`scipy.linalg.lu`); QR
decomposition via Householder reflections (:func:`scipy.linalg.qr`),
with a hand-rolled Gram-Schmidt variant kept only for the pedagogical
Householder-vs-Gram-Schmidt stability comparison; Cholesky decomposition
for SPD matrices (:func:`numpy.linalg.cholesky`); eigenvalue computation
(:func:`numpy.linalg.eigh`/:func:`numpy.linalg.eig`); power iteration and
inverse iteration for dominant/nearest eigenvalues (hand-rolled -- the
iteration itself is the pedagogical subject, with no library equivalent);
SVD (:func:`numpy.linalg.svd`); conjugate gradient and GMRES
(:func:`scipy.sparse.linalg.cg`/:func:`~scipy.sparse.linalg.gmres`);
condition number (:func:`numpy.linalg.cond`) and least-squares
(:func:`numpy.linalg.lstsq`), with a comparison utility showing why
normal equations are less stable than QR for ill-conditioned systems.
Also: Cramer's rule (:func:`numpy.linalg.slogdet`); characteristic
polynomials (:func:`numpy.poly`) and the Cayley-Hamilton theorem;
Jacobi, Gauss-Seidel, and SOR stationary iterations (hand-rolled -- no
library equivalent); the Moore-Penrose pseudoinverse
(:func:`numpy.linalg.pinv`); Gershgorin discs; Lanczos extreme
eigenpairs (:func:`scipy.sparse.linalg.eigsh`); and Hessenberg/Schur
forms (:func:`scipy.linalg.hessenberg`/:func:`~scipy.linalg.schur`).

mathematicskit's ``systems/`` classes/functions wrap these library calls in the
package's own dataclass results, docstrings, and stability commentary --
they are thin, well-tested wrappers, not reimplementations of already-
correct numerical primitives.
"""

from mathematicskit.linalg.core.base import (
    CholeskyResult,
    EigenResult,
    GershgorinResult,
    IterativeLinearSolver,
    IterativeSolveResult,
    LeastSquaresResult,
    LUResult,
    QRResult,
    SchurResult,
    SVDResult,
)
from mathematicskit.linalg.systems.cholesky import cholesky_decompose, cholesky_solve, is_symmetric_positive_definite
from mathematicskit.linalg.systems.cramer import cramer_solve
from mathematicskit.linalg.systems.eigen import eigen_general, eigen_symmetric, inverse_iteration, power_iteration
from mathematicskit.linalg.systems.gershgorin import gershgorin_discs
from mathematicskit.linalg.systems.iterative import GMRES, ConjugateGradient
from mathematicskit.linalg.systems.lanczos import lanczos_eigsh
from mathematicskit.linalg.systems.lu import lu_decompose, lu_det, lu_solve, lu_solve_system
from mathematicskit.linalg.systems.matrix_polynomial import characteristic_polynomial, matrix_polynomial
from mathematicskit.linalg.systems.pseudoinverse import penrose_residuals, pseudoinverse
from mathematicskit.linalg.systems.qr import gram_schmidt_qr, householder_qr, orthogonality_error
from mathematicskit.linalg.systems.schur import hessenberg_reduce, schur_decompose
from mathematicskit.linalg.systems.stability import condition_number_2norm, least_squares_lstsq, least_squares_normal_equations, least_squares_qr
from mathematicskit.linalg.systems.stationary import SOR, GaussSeidel, JacobiIteration, jacobi_spectral_radius, optimal_sor_omega
from mathematicskit.linalg.systems.svd import svd_decompose
from mathematicskit.linalg.utils.matrix_utils import frobenius_norm, is_symmetric, random_spd_matrix

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LUResult",
    "QRResult",
    "CholeskyResult",
    "EigenResult",
    "SVDResult",
    "IterativeSolveResult",
    "LeastSquaresResult",
    "GershgorinResult",
    "SchurResult",
    "IterativeLinearSolver",
    "lu_decompose",
    "lu_solve",
    "lu_solve_system",
    "lu_det",
    "householder_qr",
    "gram_schmidt_qr",
    "orthogonality_error",
    "cholesky_decompose",
    "cholesky_solve",
    "is_symmetric_positive_definite",
    "eigen_symmetric",
    "eigen_general",
    "power_iteration",
    "inverse_iteration",
    "svd_decompose",
    "ConjugateGradient",
    "GMRES",
    "condition_number_2norm",
    "least_squares_normal_equations",
    "least_squares_qr",
    "least_squares_lstsq",
    "cramer_solve",
    "characteristic_polynomial",
    "matrix_polynomial",
    "JacobiIteration",
    "GaussSeidel",
    "SOR",
    "jacobi_spectral_radius",
    "optimal_sor_omega",
    "pseudoinverse",
    "penrose_residuals",
    "gershgorin_discs",
    "lanczos_eigsh",
    "hessenberg_reduce",
    "schur_decompose",
    "random_spd_matrix",
    "frobenius_norm",
    "is_symmetric",
]
