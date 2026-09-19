:orphan:

Counting, Primes, and Probability
====================================

The binomial coefficient :math:`\binom{n}{k}` is the meeting point of
three mathematicskit domains that otherwise look unrelated:
:mod:`mathematicskit.combinatorics` studies it as a pure counting problem,
:mod:`mathematicskit.number_theory` studies the *prime factorizations* hiding
inside it and the totient function built from them, and
:mod:`mathematicskit.probability` uses it, unchanged, as the weight in the
binomial distribution's probability mass function.

Counting combinations
--------------------------

:func:`~mathematicskit.combinatorics.systems.counting.combinations_count`
counts the number of ways to choose 6 numbers from 20, exactly the
combinatorics behind a lottery draw:

.. code-block:: python

   from mathematicskit.combinatorics import combinations_count

   count = combinations_count(20, 6)
   print(count)
   # 38760

The same number as a probability weight
------------------------------------------------

:class:`~mathematicskit.probability.systems.discrete.Binomial`'s probability
mass function is built from exactly this same coefficient:

.. math::

   P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}

.. code-block:: python

   from mathematicskit.probability import Binomial

   binom = Binomial(n=20, p=0.5)
   print(binom.pmf(6))
   # 0.03696441650390626
   print(38760 * 0.5**6 * 0.5**14)
   # 0.03696441650390625 -- the same value, computed by hand from combinations_count's output

Primes and the totient function
--------------------------------------

:mod:`mathematicskit.number_theory` supplies the multiplicative-function
machinery this combinatorial counting eventually leans on for larger
problems -- for instance, Euler's totient function
:func:`~mathematicskit.number_theory.systems.totient.euler_totient`, which for
any prime :math:`p` always equals :math:`p-1`:

.. code-block:: python

   from mathematicskit.number_theory import sieve_of_eratosthenes, euler_totient

   primes = sieve_of_eratosthenes(50)
   print([int(p) for p in primes])
   # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

   print([euler_totient(int(p)) for p in primes])
   # [1, 2, 4, 6, 10, 12, 16, 18, 22, 28, 30, 36, 40, 42, 46] -- each exactly p - 1

Every prime :math:`p` has :math:`\varphi(p) = p-1` precisely because
every integer from 1 to :math:`p-1` is automatically coprime to a prime
-- the same fact (via Euler's theorem, :math:`a^{\varphi(n)}\equiv1
\pmod n`) that RSA cryptography's key generation in
:doc:`/history/number_theory_breakthroughs` depends on, and, via
Fermat's Little Theorem's special case, the same fact
:func:`mathematicskit.number_theory.systems.primality.is_prime_miller_rabin`
tests probabilistically for much larger candidate primes than
:func:`sieve_of_eratosthenes` could ever enumerate directly.

See Also
--------

- :doc:`/api/combinatorics`
- :doc:`/api/number_theory`
- :doc:`/api/probability`
- :doc:`/history/combinatorics_breakthroughs`
- :doc:`/history/number_theory_breakthroughs`