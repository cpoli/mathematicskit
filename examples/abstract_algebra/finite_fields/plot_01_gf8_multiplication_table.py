r"""
GF(8): finding an irreducible polynomial and building the field
========================================================================

Finds a degree-3 irreducible polynomial over GF(2), builds the field
GF(2^3) = GF(8) from it, and verifies every nonzero element has a
multiplicative inverse.
"""

# %%
from mathematicskit.abstract_algebra import GF, Polynomial, find_irreducible_polynomial, is_irreducible

# %%
# Find the defining irreducible polynomial
# -----------------------------------------------------

irreducible = find_irreducible_polynomial(2, 3)
print(f"irreducible polynomial over GF(2): coefficients {irreducible.coeffs} (degree {irreducible.degree})")
print(f"confirmed irreducible: {is_irreducible(irreducible)}")

# %%
# Build GF(8) and verify every nonzero element has an inverse
# -------------------------------------------------------------------

field = GF(2, 3, irreducible=irreducible)
zero = Polynomial([0], modulus=2)
one = Polynomial([1], modulus=2)

print(f"\nGF(8) has {field.order} elements")
for element in field.elements():
    if element == zero:
        continue
    inverse = field.inverse(element)
    check = field.multiply(element, inverse)
    print(f"  {element.coeffs} * {inverse.coeffs} = {check.coeffs} (should be [1]: {check == one})")
