r"""
Noether's first isomorphism theorem
=========================================

Checks :math:`|G| = |\ker\varphi| \cdot |\operatorname{im}\varphi|`
for every homomorphism :math:`x \mapsto ax` from Z_12 to itself, and
shows that the sign map on S_4 has the alternating group A_4 as its
kernel.
"""

# %%
from mathematicskit.abstract_algebra import (
    CyclicGroup,
    PermutationGroup,
    analyze_homomorphism,
    is_normal_subgroup,
    quotient_group,
)

# %%
# Multiplication maps on Z_12
# -----------------------------------------------------

z12 = CyclicGroup(12)
for a in range(12):
    result = analyze_homomorphism(z12, z12, lambda x, a=a: (a * x) % 12)
    q = quotient_group(z12, result.kernel)
    print(f"x -> {a:2d}x: |ker| = {len(result.kernel):2d}, |im| = {len(result.image):2d}, |G/ker| = {q.order:2d}")

# %%
# The sign homomorphism S_4 -> Z_2
# -----------------------------------------------------


def sign(p):
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j]) % 2


s4 = PermutationGroup(4)
result = analyze_homomorphism(s4, CyclicGroup(2), sign)
print(f"\nsign is a homomorphism: {result.is_homomorphism}")
print(f"kernel (A_4) has order {len(result.kernel)} and is normal: {is_normal_subgroup(s4, result.kernel)}")
