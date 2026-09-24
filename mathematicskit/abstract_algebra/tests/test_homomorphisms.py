"""Tests for group homomorphisms and the first isomorphism theorem."""

from mathematicskit.abstract_algebra.systems.groups import CyclicGroup, PermutationGroup
from mathematicskit.abstract_algebra.systems.homomorphisms import (
    analyze_homomorphism,
    homomorphism_image,
    homomorphism_kernel,
    is_homomorphism,
)
from mathematicskit.abstract_algebra.systems.structure import is_normal_subgroup, quotient_group


def _sign(p):
    inversions = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return inversions % 2  # as an element of Z_2


def test_reduction_mod_4_is_a_homomorphism_from_z12():
    assert is_homomorphism(CyclicGroup(12), CyclicGroup(4), lambda a: a % 4)
    assert not is_homomorphism(CyclicGroup(12), CyclicGroup(5), lambda a: a % 5)


def test_sign_map_kernel_is_the_alternating_group():
    s4 = PermutationGroup(4)
    z2 = CyclicGroup(2)
    assert is_homomorphism(s4, z2, _sign)
    kernel = homomorphism_kernel(s4, z2, _sign)
    assert len(kernel) == 12
    assert is_normal_subgroup(s4, kernel)
    assert homomorphism_image(s4, z2, _sign) == [0, 1]


def test_first_isomorphism_theorem_orders():
    g, h = CyclicGroup(12), CyclicGroup(12)
    phi = lambda a: (3 * a) % 12  # noqa: E731
    result = analyze_homomorphism(g, h, phi)
    assert result.is_homomorphism
    assert len(result.kernel) * len(result.image) == g.order
    assert quotient_group(g, result.kernel).order == len(result.image)
