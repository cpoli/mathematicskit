Breakthroughs in Number Theory
==============================


.. include:: /_generated/nav/number_theory.rst

.. epigraph::

   "Mathematics is the queen of the sciences, and number theory is the
   queen of mathematics."
   -- attributed to Carl Friedrich Gauss by Wolfgang Sartorius von
   Waltershausen, 1856

Number theory is the oldest continuously studied branch of mathematics.
It still yields deep theorems and, unexpectedly, some of the most
consequential technology of the digital age. This chronology traces the
ideas behind :mod:`mathematicskit.number_theory`, from Euclid's original
algorithm to the discovery that its arithmetic underlies modern
public-key cryptography.

.. contents:: Timeline
   :local:
   :depth: 1

c. 300 BCE -- Euclid's Algorithm
--------------------------------

Book VII of Euclid's *Elements* finds the greatest common measure of two
numbers by repeated subtraction. It is the same greatest-common-divisor
algorithm that still bears his name, and a strong candidate for the
oldest algorithm still in everyday use. The *extended* algorithm also
tracks the integer coefficients that express the gcd as a combination
of the two inputs. It is nearly as old: Aryabhata's *kuttaka* method of
499 CE solves the same problem. In Europe, Claude-Gaspard Bachet de
Méziriac stated the resulting identity for integers in 1624, and Étienne
Bézout, whose name it now carries, generalized it to polynomials in
1779.

.. math::

   \gcd(a, b) = \gcd(b, a \bmod b), \qquad ax + by = \gcd(a, b)

*Implementation:* :func:`mathematicskit.number_theory.systems.modular_arithmetic.extended_gcd`
implements the extended algorithm, unwinding Euclid's recursion to
recover Bézout's coefficients;
:func:`~mathematicskit.number_theory.systems.modular_arithmetic.mod_inverse`
uses it to compute modular inverses.

*References:* Euclid, *Elements*, Book VII, Propositions 1-2 (c. 300
BCE), as translated in T. L. Heath, *The Thirteen Books of Euclid's
Elements*, 2nd ed. (Cambridge: Cambridge University Press, 1926).

.. minigallery:: ../../examples/number_theory/modular_arithmetic/plot_01_toy_rsa.py

c. 250 CE -- Diophantus and Indeterminate Equations
---------------------------------------------------

Diophantus of Alexandria's *Arithmetica* systematically studied
equations to be solved in whole numbers or rationals, a class of
problem now called "Diophantine" in his honor. A linear Diophantine
equation :math:`ax+by=c` has an integer solution exactly when
:math:`\gcd(a,b)` divides :math:`c`, which follows directly from
Euclid's algorithm (above). The much harder quadratic case,
:math:`x^2 - Dy^2 = 1`, would not be fully understood for another
millennium and a half.

*Implementation:* :func:`mathematicskit.number_theory.systems.diophantine.solve_linear_diophantine`
solves the linear case with the Bézout coefficients from the extended
Euclidean algorithm.

*References:* Diophantus of Alexandria, *Arithmetica* (c. 250 CE), as
reconstructed and translated in T. L. Heath, *Diophantus of Alexandria:
A Study in the History of Greek Algebra*, 2nd ed. (Cambridge: Cambridge
University Press, 1910).

.. minigallery:: ../../examples/number_theory/diophantine/plot_01_pell_and_linear.py

1613-1737 -- Continued Fractions and Best Rational Approximation
----------------------------------------------------------------

Pietro Cataldi's 1613 treatise on square roots gave the first published
continued-fraction expansion, approximating a square root by an
infinite nested sequence of reciprocals. Leonhard Euler's "De
Fractionibus Continuis," presented in 1737, gave continued fractions
their first systematic theory. Joseph-Louis Lagrange later proved the
property that makes them useful far beyond square roots: each
convergent :math:`p_k/q_k`, obtained by truncating the expansion, is the
best rational approximation to the original number among all fractions
with an equal or smaller denominator.

.. math::

   x = a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{a_3 + \dots}}}

*Implementation:* :func:`mathematicskit.number_theory.systems.continued_fractions.continued_fraction_expansion`
computes the expansion and its convergents with the standard recurrence
:math:`p_k = a_k p_{k-1} + p_{k-2}`.
:func:`~mathematicskit.number_theory.systems.continued_fractions.best_rational_approximation`
uses it to find the best rational approximation under a denominator
bound, reproducing the classical approximation :math:`\pi \approx
355/113`. Applied to :math:`\sqrt D`, the same expansion is the method
for solving Pell's equation (below).

*References:* P. A. Cataldi, *Trattato del modo brevissimo di trovar la
radice quadra delli numeri* (Bologna, 1613); L. Euler, "De Fractionibus
Continuis Dissertatio," Commentarii Academiae Scientiarum
Petropolitanae 9 (1744), 98-137 (presented to the St. Petersburg
Academy in 1737).

.. minigallery:: ../../examples/number_theory/continued_fractions/plot_01_approximating_pi.py

1657-1768 -- Fermat, Euler, and Pell's Equation
-----------------------------------------------

In 1657 Pierre de Fermat challenged his contemporaries to solve
:math:`x^2 - Dy^2 = 1` in integers. He knew that, despite the simple
form, the solutions can be enormous for innocent-looking values of
:math:`D`: for :math:`D=61` the smallest solution has
:math:`x=1766319049`. William Brouncker answered the challenge with a
general method in 1657-58, and Brahmagupta and Bhaskara II had solved
many cases centuries earlier. The name "Pell's equation" is a
misattribution: Leonhard Euler wrongly credited John Pell with
Brouncker's method. Euler connected the method to the continued
fraction of :math:`\sqrt D` (above), and in 1768 Joseph-Louis Lagrange
proved that it always terminates and yields every solution.

*Implementation:* :func:`mathematicskit.number_theory.systems.diophantine.solve_pell_equation`
implements this continued-fraction algorithm and reproduces the
:math:`D=61` fundamental solution.

*References:* J.-L. Lagrange, "Solution d'un problème d'arithmétique,"
Miscellanea Taurinensia 4 (1766-69), 19-99.

.. minigallery:: ../../examples/number_theory/diophantine/plot_01_pell_and_linear.py

1763 -- Euler's Totient Function
--------------------------------

While generalizing Fermat's little theorem, Leonhard Euler introduced
the function :math:`\varphi(n)` that counts the integers up to
:math:`n` that are coprime to it. The symbol :math:`\varphi` is Carl
Friedrich Gauss's, and James Joseph Sylvester later named it the
"totient." Euler's theorem, :math:`a^{\varphi(n)} \equiv 1 \pmod n` for
:math:`\gcd(a,n)=1`, generalizes Fermat's little theorem, which is the
special case of a prime :math:`n` with :math:`\varphi(n)=n-1`. Two
centuries later, it is exactly the fact that RSA public-key
cryptography relies on to guarantee that decryption undoes encryption.

*Implementation:* :func:`mathematicskit.number_theory.systems.totient.euler_totient`
computes this function from the prime factorization, alongside the
related Möbius function
(:func:`~mathematicskit.number_theory.systems.totient.mobius`) and
divisor-sum function
(:func:`~mathematicskit.number_theory.systems.totient.divisor_sum`).

*References:* L. Euler, "Theoremata arithmetica nova methodo demonstrata,"
Novi Commentarii Academiae Scientiarum Petropolitanae 8 (1763), 74-104.

.. minigallery:: ../../examples/number_theory/totient/plot_01_perfect_numbers.py

1801 -- Gauss's Disquisitiones Arithmeticae
-------------------------------------------

Carl Friedrich Gauss published the *Disquisitiones Arithmeticae* at 24.
It made congruence arithmetic ("clock arithmetic") the central
organizing language of number theory, and introduced the notation
:math:`a \equiv b \pmod m` that is still used today. Among its results
is a general treatment of the Chinese remainder theorem, which combines
several congruences with pairwise coprime moduli into a single
equivalent one. The theorem is named for its Chinese origin: its oldest
known instance is a puzzle in the *Sunzi Suanjing*, written between the
3rd and 5th centuries.

*Implementation:* :func:`mathematicskit.number_theory.systems.crt.chinese_remainder_theorem`
implements this combination, and
:func:`~mathematicskit.number_theory.systems.modular_arithmetic.fast_mod_pow`
implements the fast modular exponentiation that makes congruence
arithmetic with large moduli practical.

*References:* C. F. Gauss, *Disquisitiones Arithmeticae* (Leipzig:
Fleischer, 1801), Sections I-II, especially Articles 32-36 (Chinese
remainder theorem).

.. minigallery:: ../../examples/number_theory/crt/plot_01_sunzi_problem.py

1973-1978 -- RSA and Public-Key Cryptography
--------------------------------------------

Ronald Rivest, Adi Shamir, and Leonard Adleman's 1978 paper turned
Leonhard Euler's 1763 theorem into a working cryptosystem. Choose two
large primes, and publish their product together with an exponent. A
message encrypted with that public exponent can be decrypted only by
the holder of the matching private exponent, because recovering it
requires factoring the product -- which, as far as anyone has publicly
shown, is computationally infeasible. The scheme is built from the
totient and modular-inverse machinery that Euler and Carl Friedrich
Gauss had developed two centuries earlier for entirely different
reasons. Clifford Cocks at Britain's GCHQ had derived an equivalent
scheme in 1973, but it stayed classified until 1997.

*Implementation:* mathematicskit's toy demonstration combines
:func:`mathematicskit.number_theory.systems.totient.euler_totient`,
:func:`~mathematicskit.number_theory.systems.modular_arithmetic.mod_inverse`,
and :func:`~mathematicskit.number_theory.systems.modular_arithmetic.fast_mod_pow`
to run an RSA key-generation, encryption, and decryption round trip. Its
small primes are illustrative, not cryptographically secure.

*References:* R. L. Rivest, A. Shamir, and L. Adleman, "A Method for
Obtaining Digital Signatures and Public-Key Cryptosystems,"
Communications of the ACM 21(2) (1978), 120-126.

.. minigallery:: ../../examples/number_theory/modular_arithmetic/plot_01_toy_rsa.py

1976-1980 -- Miller, Rabin, and Primality Testing
-------------------------------------------------

Gary Miller's 1976 paper gave a deterministic primality test whose
correctness depends on an unproven conjecture, the generalized Riemann
hypothesis. Michael Rabin's 1980 paper turned Miller's test into an
unconditionally correct *probabilistic* one by choosing the test
witnesses at random. The price is a chance of error that shrinks
exponentially with the number of witnesses tried. Unlike trial
division, the test is fast enough to certify primes hundreds of digits
long, exactly the size RSA key generation needs.

*Implementation:* :func:`mathematicskit.number_theory.systems.primality.is_prime_miller_rabin`
implements this randomized test. The test suite checks it over a wide
range against the much slower but always-certain
:func:`~mathematicskit.number_theory.systems.primality.is_prime_trial_division`.
:func:`~mathematicskit.number_theory.systems.primality.sieve_of_eratosthenes`
implements the much older (c. 240 BCE) sieve of Eratosthenes, which
generates every prime up to a limit at once.

*References:* G. L. Miller, "Riemann's Hypothesis and Tests for
Primality," Journal of Computer and System Sciences 13(3) (1976),
300-317; M. O. Rabin, "Probabilistic Algorithm for Testing Primality,"
Journal of Number Theory 12(1) (1980), 128-138.

.. minigallery:: ../../examples/number_theory/primality/plot_01_prime_counting.py

See Also
--------

- :doc:`/api/number_theory`
- :doc:`/history/combinatorics_breakthroughs`
- :doc:`/history/abstract_algebra_breakthroughs`
