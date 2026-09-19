"""Concrete linear-algebra algorithms: decompositions, eigensolvers, SVD,
iterative solvers, and stability comparisons.
"""

from mathematicskit.linalg.systems.cholesky import cholesky_decompose, cholesky_solve, is_symmetric_positive_definite
from mathematicskit.linalg.systems.eigen import eigen_general, eigen_symmetric, inverse_iteration, power_iteration
from mathematicskit.linalg.systems.iterative import GMRES, ConjugateGradient
from mathematicskit.linalg.systems.lu import lu_decompose, lu_det, lu_solve, lu_solve_system
from mathematicskit.linalg.systems.qr import gram_schmidt_qr, householder_qr, orthogonality_error
from mathematicskit.linalg.systems.stability import condition_number_2norm, least_squares_lstsq, least_squares_normal_equations, least_squares_qr
from mathematicskit.linalg.systems.svd import svd_decompose

__all__ = [
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
]
