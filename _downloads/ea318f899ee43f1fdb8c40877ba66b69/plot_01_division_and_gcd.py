r"""
Polynomial long division and the Euclidean algorithm
============================================================

Divides two polynomials with remainder, and finds their greatest
common divisor via the polynomial Euclidean algorithm -- exactly the
integer Euclidean algorithm with polynomial division in place of
integer division.
"""

# %%
from mathkit.abstract_algebra import Polynomial, poly_divmod, poly_gcd, poly_mul

# %%
# Divide x^3 - 1 by x - 1
# -----------------------------------------------------

p = Polynomial([-1, 0, 0, 1])  # x^3 - 1
q = Polynomial([-1, 1])  # x - 1
quotient, remainder = poly_divmod(p, q)
print(f"(x^3 - 1) / (x - 1) = quotient {quotient.coeffs}, remainder {remainder.coeffs}")
print(f"check: quotient * (x-1) + remainder = {(poly_mul(quotient, q) + remainder).coeffs}")

# %%
# GCD of x^4 - 1 and x^6 - 1
# -----------------------------------------------------

a = Polynomial([-1, 0, 0, 0, 1])  # x^4 - 1
b = Polynomial([-1, 0, 0, 0, 0, 0, 1])  # x^6 - 1
g = poly_gcd(a, b)
print(f"\ngcd(x^4 - 1, x^6 - 1) has degree {g.degree}, coefficients {g.coeffs}")
