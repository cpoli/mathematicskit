"""Tests for elliptic-curve arithmetic over GF(p) and Lenstra's factorization."""

import math

import pytest

from mathematicskit.number_theory.systems.elliptic_curves import EllipticCurve, lenstra_ecm


@pytest.fixture
def curve():
    return EllipticCurve(2, 3, 97)


def test_group_axioms_on_all_points(curve):
    points = curve.points() + [None]
    sample = points[::5]
    for p in sample:
        assert curve.add(p, None) == p
        assert curve.add(p, curve.negate(p)) is None
        for q in sample:
            assert curve.add(p, q) == curve.add(q, p)
            assert curve.is_on_curve(curve.add(p, q))
            for r in sample[::3]:
                assert curve.add(curve.add(p, q), r) == curve.add(p, curve.add(q, r))


def test_hasse_bound_and_lagrange():
    for p in (101, 211, 499):
        for a, b in [(1, 1), (2, 7), (0, 5)]:
            E = EllipticCurve(a, b, p)
            n = E.order()
            assert abs(n - (p + 1)) <= 2 * math.sqrt(p)
            for point in E.points()[::17]:
                assert n % E.point_order(point) == 0
                assert E.multiply(n, point) is None


def test_multiply_matches_repeated_addition(curve):
    p = (3, 6)
    q = None
    for k in range(1, 12):
        q = curve.add(q, p)
        assert curve.multiply(k, p) == q
    assert curve.multiply(0, p) is None
    assert curve.multiply(-3, p) == curve.negate(curve.multiply(3, p))


def test_supersingular_curve_order():
    # y^2 = x^3 + x over GF(p) with p = 3 (mod 4) has exactly p + 1 points.
    for p in (7, 11, 19, 23, 43):
        assert EllipticCurve(1, 0, p).order() == p + 1


def test_invalid_curves():
    with pytest.raises(ValueError):
        EllipticCurve(0, 0, 97)
    with pytest.raises(ValueError):
        EllipticCurve(1, 1, 3)


def test_ecm_factors_semiprimes():
    for p, q in [(1009, 1013), (65537, 1000003), (999983, 1000000007)]:
        result = lenstra_ecm(p * q, seed=1)
        assert {result.factor, result.cofactor} == {p, q}
        assert result.curves >= 1 and len(result.curve) == 4


def test_ecm_finds_small_factor_of_large_number():
    n = 1000000007 * 998244353 * 10007
    result = lenstra_ecm(n, b1=500, seed=2)
    assert 1 < result.factor < n and n % result.factor == 0


def test_ecm_rejects_primes_and_small_input():
    with pytest.raises(ValueError):
        lenstra_ecm(1000003)
    with pytest.raises(ValueError):
        lenstra_ecm(3)
    assert lenstra_ecm(22).factor == 2
