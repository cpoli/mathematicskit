"""mathkit.special_functions: special functions and transforms.

The gamma and beta functions (``scipy.special.gamma``/``beta``); Bessel
functions of the first/second kind (``scipy.special.jv``/``yv``);
orthogonal polynomial families (Legendre, Chebyshev, Hermite, Laguerre)
via ``numpy.polynomial``, with orthogonality verified numerically in
tests; and the discrete Fourier transform via ``numpy.fft``, plus a
from-scratch radix-2 FFT kept specifically as a pedagogical comparison
against a naive :math:`O(n^2)` DFT and ``numpy.fft`` -- the one
sub-module in this domain where hand-rolling is the point.
"""

from mathkit.special_functions.core.base import FFTComparisonResult
from mathkit.special_functions.systems.bessel import bessel_first_kind, bessel_second_kind
from mathkit.special_functions.systems.fourier_transform import compare_fft_methods, dft_naive, fft_numpy, fft_radix2
from mathkit.special_functions.systems.gamma_beta import beta_function, gamma_function, log_gamma_function
from mathkit.special_functions.systems.orthogonal_polynomials import chebyshev_polynomial, hermite_polynomial, laguerre_polynomial, legendre_polynomial
from mathkit.special_functions.utils.orthogonality import inner_product

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FFTComparisonResult",
    "gamma_function",
    "log_gamma_function",
    "beta_function",
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
