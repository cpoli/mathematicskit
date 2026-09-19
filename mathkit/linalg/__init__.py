"""mathkit.linalg: numerical linear algebra, built on numpy.linalg/scipy.linalg/scipy.sparse.linalg.

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

mathkit's ``systems/`` classes/functions wrap these library calls in the
package's own dataclass results, docstrings, and stability commentary --
they are thin, well-tested wrappers, not reimplementations of already-
correct numerical primitives.
"""

from mathkit.linalg.core.base import (
    CholeskyResult,
    EigenResult,
    IterativeLinearSolver,
    IterativeSolveResult,
    LeastSquaresResult,
    LUResult,
    QRResult,
    SVDResult,
)
from mathkit.linalg.systems.cholesky import cholesky_decompose, cholesky_solve, is_symmetric_positive_definite
from mathkit.linalg.systems.eigen import eigen_general, eigen_symmetric, inverse_iteration, power_iteration
from mathkit.linalg.systems.iterative import GMRES, ConjugateGradient
from mathkit.linalg.systems.lu import lu_decompose, lu_det, lu_solve, lu_solve_system
from mathkit.linalg.systems.qr import gram_schmidt_qr, householder_qr, orthogonality_error
from mathkit.linalg.systems.stability import condition_number_2norm, least_squares_lstsq, least_squares_normal_equations, least_squares_qr
from mathkit.linalg.systems.svd import svd_decompose
from mathkit.linalg.utils.matrix_utils import frobenius_norm, is_symmetric, random_spd_matrix

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
    "random_spd_matrix",
    "frobenius_norm",
    "is_symmetric",
]
