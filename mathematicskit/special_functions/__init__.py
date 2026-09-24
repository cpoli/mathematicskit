"""mathematicskit.special_functions: special functions and transforms.

The gamma and beta functions (``scipy.special.gamma``/``beta``) with
Stirling's asymptotic series; Bessel, Airy, and Mathieu functions;
elliptic integrals, the arithmetic-geometric mean, and Jacobi elliptic
functions; Gauss's and Kummer's hypergeometric functions; the error
function and Fresnel integrals; the Riemann zeta function and Euler's
prime product; the Lambert W function; orthogonal polynomial families
(Legendre, Chebyshev, Hermite, Laguerre) via ``numpy.polynomial``, with
orthogonality verified numerically in tests; and the discrete Fourier
transform via ``numpy.fft``, plus a from-scratch radix-2 FFT kept
specifically as a pedagogical comparison against a naive :math:`O(n^2)`
DFT and ``numpy.fft`` -- the one sub-module in this domain where
hand-rolling is the point.
"""

from mathematicskit.special_functions.core.base import AiryResult, FFTComparisonResult, FresnelResult, JacobiEllipticResult
from mathematicskit.special_functions.systems.airy import airy_functions
from mathematicskit.special_functions.systems.bessel import bessel_first_kind, bessel_second_kind
from mathematicskit.special_functions.systems.elliptic import (
    arithmetic_geometric_mean,
    complete_elliptic_integral_first_kind,
    complete_elliptic_integral_second_kind,
    jacobi_elliptic_functions,
)
from mathematicskit.special_functions.systems.error_functions import complementary_error_function, error_function, fresnel_integrals
from mathematicskit.special_functions.systems.fourier_transform import compare_fft_methods, dft_naive, fft_numpy, fft_radix2
from mathematicskit.special_functions.systems.gamma_beta import beta_function, gamma_function, log_gamma_function, stirling_factorial, stirling_log_gamma
from mathematicskit.special_functions.systems.hypergeometric import confluent_hypergeometric_1f1, hypergeometric_2f1
from mathematicskit.special_functions.systems.lambert_w import lambert_w
from mathematicskit.special_functions.systems.mathieu import mathieu_characteristic_a, mathieu_characteristic_b, mathieu_even, mathieu_odd
from mathematicskit.special_functions.systems.orthogonal_polynomials import chebyshev_polynomial, hermite_polynomial, laguerre_polynomial, legendre_polynomial
from mathematicskit.special_functions.systems.zeta import euler_product, riemann_zeta
from mathematicskit.special_functions.utils.orthogonality import inner_product

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FFTComparisonResult",
    "JacobiEllipticResult",
    "AiryResult",
    "FresnelResult",
    "gamma_function",
    "log_gamma_function",
    "beta_function",
    "stirling_factorial",
    "stirling_log_gamma",
    "lambert_w",
    "arithmetic_geometric_mean",
    "complete_elliptic_integral_first_kind",
    "complete_elliptic_integral_second_kind",
    "jacobi_elliptic_functions",
    "hypergeometric_2f1",
    "confluent_hypergeometric_1f1",
    "error_function",
    "complementary_error_function",
    "fresnel_integrals",
    "airy_functions",
    "riemann_zeta",
    "euler_product",
    "mathieu_characteristic_a",
    "mathieu_characteristic_b",
    "mathieu_even",
    "mathieu_odd",
    "bessel_first_kind",
    "bessel_second_kind",
    "legendre_polynomial",
    "chebyshev_polynomial",
    "hermite_polynomial",
    "laguerre_polynomial",
    "inner_product",
    "dft_naive",
    "fft_radix2",
    "fft_numpy",
    "compare_fft_methods",
]
