"""Concrete number-theoretic algorithms."""

from mathematicskit.number_theory.systems.continued_fractions import best_rational_approximation, continued_fraction_expansion
from mathematicskit.number_theory.systems.crt import chinese_remainder_theorem
from mathematicskit.number_theory.systems.diophantine import solve_linear_diophantine, solve_pell_equation
from mathematicskit.number_theory.systems.modular_arithmetic import extended_gcd, fast_mod_pow, mod_inverse
from mathematicskit.number_theory.systems.primality import is_prime_miller_rabin, is_prime_trial_division, sieve_of_eratosthenes
from mathematicskit.number_theory.systems.totient import divisor_sum, euler_totient, mobius, prime_factorization

__all__ = [
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
]
