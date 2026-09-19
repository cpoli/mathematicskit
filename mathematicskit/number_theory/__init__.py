"""mathematicskit.number_theory: elementary and computational number theory.

Every algorithm here is hand-rolled from its textbook definition --
exact-integer arithmetic has no ``numpy``/``scipy`` equivalent. The
extended Euclidean algorithm and modular inverses; fast modular
exponentiation; primality testing (trial division, Miller-Rabin) and
prime generation (sieve of Eratosthenes); the Chinese Remainder Theorem;
continued-fraction expansion and best rational approximations; Euler's
totient function and other multiplicative number-theoretic functions
(Mobius, divisor-sum); and linear and Pell Diophantine equation solvers.
"""

from mathematicskit.number_theory.core.base import BezoutResult, ContinuedFractionResult, CRTResult, LinearDiophantineResult, PellResult
from mathematicskit.number_theory.systems.continued_fractions import best_rational_approximation, continued_fraction_expansion
from mathematicskit.number_theory.systems.crt import chinese_remainder_theorem
from mathematicskit.number_theory.systems.diophantine import solve_linear_diophantine, solve_pell_equation
from mathematicskit.number_theory.systems.modular_arithmetic import extended_gcd, fast_mod_pow, mod_inverse
from mathematicskit.number_theory.systems.primality import is_prime_miller_rabin, is_prime_trial_division, sieve_of_eratosthenes
from mathematicskit.number_theory.systems.totient import divisor_sum, euler_totient, mobius, prime_factorization
from mathematicskit.number_theory.utils.gcd_lcm import gcd, lcm

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BezoutResult",
    "ContinuedFractionResult",
    "LinearDiophantineResult",
    "PellResult",
    "CRTResult",
    "extended_gcd",
    "mod_inverse",
    "fast_mod_pow",
    "is_prime_trial_division",
    "is_prime_miller_rabin",
    "sieve_of_eratosthenes",
    "chinese_remainder_theorem",
    "continued_fraction_expansion",
    "best_rational_approximation",
    "prime_factorization",
    "euler_totient",
    "mobius",
    "divisor_sum",
    "solve_linear_diophantine",
    "solve_pell_equation",
    "gcd",
    "lcm",
]
