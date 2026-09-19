"""Tests for Euler's totient and other multiplicative functions against
closed-form/known results."""

from mathkit.number_theory.systems.totient import divisor_sum, euler_totient, mobius, prime_factorization


def test_prime_factorization_matches_known_values():
    assert prime_factorization(360) == {2: 3, 3: 2, 5: 1}
    assert prime_factorization(1) == {}
    assert prime_factorization(17) == {17: 1}


def test_prime_factorization_reconstructs_n():
    for n in (2, 3, 100, 360, 9999):
        factors = prime_factorization(n)
        product = 1
        for p, e in factors.items():
            product *= p**e
        assert product == n


def test_euler_totient_of_prime_is_p_minus_1():
    for p in (2, 3, 5, 7, 101):
        assert euler_totient(p) == p - 1


def test_euler_totient_matches_brute_force():
    for n in range(1, 100):
        expected = sum(1 for k in range(1, n + 1) if _gcd(k, n) == 1)
        assert euler_totient(n) == expected


def test_euler_totient_is_multiplicative_for_coprime_arguments():
    # phi(9) * phi(5) should equal phi(45) since gcd(9,5)=1.
    assert euler_totient(9) * euler_totient(5) == euler_totient(45)


def test_mobius_known_sequence():
    assert [mobius(k) for k in range(1, 11)] == [1, -1, -1, 0, -1, 1, -1, 0, 0, 1]


def test_perfect_numbers_have_divisor_sum_equal_to_twice_themselves():
    for n in (6, 28, 496):
        assert divisor_sum(n) == 2 * n


def test_divisor_sum_power_zero_counts_divisors():
    assert divisor_sum(12, power=0) == 6  # 1, 2, 3, 4, 6, 12


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a
