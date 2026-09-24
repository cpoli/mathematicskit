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

1640-1749 -- Fermat's Two-Squares Theorem
-----------------------------------------

In a letter to Marin Mersenne dated Christmas Day 1640, Pierre de Fermat
claimed that an odd prime is a sum of two squares exactly when it
leaves remainder 1 on division by 4: :math:`5 = 1^2 + 2^2` and
:math:`13 = 2^2 + 3^2`, but no prime :math:`p \equiv 3 \pmod 4` can be
written this way. Fermat, as usual, gave no proof. Leonhard Euler
supplied the first one after years of effort, announcing it in 1749 and
publishing it in 1760. The theorem extends to every integer: :math:`n`
is a sum of two squares exactly when each prime :math:`q \equiv 3
\pmod 4` divides it to an even power. Proofs have kept coming ever
since, down to Don Zagier's famous "one-sentence proof" of 1990.

.. math::

   p = a^2 + b^2 \iff p = 2 \ \text{or}\ p \equiv 1 \pmod 4

*Implementation:* :func:`mathematicskit.number_theory.systems.sums_of_squares.sum_of_two_squares`
finds a representation :math:`n = a^2 + b^2` by searching
:math:`a \le \sqrt{n/2}`, or reports that none exists. The test suite
checks Fermat's criterion on every prime below 2000 and the general
criterion on every :math:`n < 1500`.

*References:* P. de Fermat, letter to M. Mersenne, 25 December 1640, in
*Œuvres de Fermat*, vol. 2, ed. P. Tannery and C. Henry (Paris:
Gauthier-Villars, 1894); L. Euler, "Demonstratio theorematis Fermatiani
omnem numerum primum formae 4n+1 esse summam duorum quadratorum," Novi
Commentarii Academiae Scientiarum Petropolitanae 5 (1760), 3-13;
D. Zagier, "A One-Sentence Proof That Every Prime :math:`p \equiv 1
\pmod 4` Is a Sum of Two Squares," American Mathematical Monthly 97(2)
(1990), 144.

.. minigallery:: ../../examples/number_theory/sums_of_squares/plot_01_fermat_two_squares.py

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

1735-1737 -- The Basel Problem and Euler's Product Formula
----------------------------------------------------------

The Basel problem asked for the exact value of :math:`\sum 1/n^2`; it
had defeated Pietro Mengoli, Gottfried Leibniz, and the Bernoulli
brothers. In 1735 the 28-year-old Leonhard Euler announced the answer,
:math:`\pi^2/6`. Two years later, in "Variae observationes circa series
infinitas," he found a far deeper identity: the same kind of sum can be
rewritten as a product over the primes. Expanding each factor as a
geometric series and multiplying out produces every term :math:`1/n^s`
exactly once, which is unique prime factorization in analytic form. At
:math:`s = 1` the sum is the divergent harmonic series, so the product
must diverge too. That gives a new proof that there are infinitely
many primes, and the first link between the primes and analysis.
Bernhard Riemann's 1859 study of this function of a complex variable
:math:`s`, now the Riemann zeta function, founded analytic number
theory.

.. math::

   \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
            = \prod_{p\ \text{prime}} \frac{1}{1 - p^{-s}}, \qquad
   \zeta(2) = \frac{\pi^2}{6}

*Implementation:* :func:`mathematicskit.number_theory.systems.zeta.euler_product`
evaluates the product over primes up to a bound. The test suite checks
it against :func:`scipy.special.zeta` and :math:`\pi^2/6`, and at
:math:`s = 1` against Mertens' asymptotic :math:`e^{\gamma}\ln N`.

*References:* L. Euler, "De summis serierum reciprocarum," Commentarii
Academiae Scientiarum Petropolitanae 7 (1740), 123-134 (presented
1735); L. Euler, "Variae observationes circa series infinitas,"
Commentarii Academiae Scientiarum Petropolitanae 9 (1744), 160-188
(presented 1737).

.. minigallery:: ../../examples/number_theory/zeta/plot_01_basel_euler_product.py

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

1770 -- Lagrange's Four-Square Theorem
--------------------------------------

Diophantus's *Arithmetica* already seems to assume that every positive
integer is a sum of four squares, and Claude-Gaspard Bachet de Méziriac
stated the claim explicitly in his 1621 edition of the book. Fermat
said he had a proof, and Euler worked on the problem for four decades
without finishing it. Joseph-Louis Lagrange completed the proof in 1770,
building on Euler's identity that a product of two sums of four squares
is again a sum of four squares. Three squares are not always enough:
Adrien-Marie Legendre showed in 1798 that the numbers
:math:`4^k(8m+7)`, such as 7, 15, and 28, need all four. In 1829 Carl
Jacobi counted the representations exactly.

.. math::

   n = a^2 + b^2 + c^2 + d^2, \qquad
   r_4(n) = 8 \sum_{d \mid n,\ 4 \nmid d} d

*Implementation:* :func:`mathematicskit.number_theory.systems.sums_of_squares.sum_of_four_squares`
finds a representation :math:`a \ge b \ge c \ge d` for any
:math:`n \ge 0`. The test suite checks every :math:`n < 3000`, and the
gallery example sorts :math:`n < 10{,}000` by the fewest squares each
one needs.

*References:* J.-L. Lagrange, "Démonstration d'un théorème
d'arithmétique," Nouveaux Mémoires de l'Académie Royale des Sciences et
Belles-Lettres de Berlin (1770), 123-133; C. G. J. Jacobi, *Fundamenta
Nova Theoriae Functionum Ellipticarum* (Königsberg: Borntraeger, 1829).

.. minigallery:: ../../examples/number_theory/sums_of_squares/plot_02_lagrange_four_squares.py

1796 -- Gauss and Quadratic Reciprocity
---------------------------------------

Which numbers are perfect squares modulo a prime :math:`p`? Leonhard
Euler and Adrien-Marie Legendre discovered a surprising symmetry: for
distinct odd primes :math:`p` and :math:`q`, whether :math:`p` is a
square modulo :math:`q` determines whether :math:`q` is a square modulo
:math:`p`. Legendre's 1785 attempt at a proof was incomplete. On 8
April 1796, aged 18, Carl Friedrich Gauss found the first complete
proof, and he published it in the *Disquisitiones Arithmeticae* (1801)
as the "fundamental theorem." He went on to give eight proofs in all,
and more than 200 are known today. The law also turns the Legendre
symbol into a fast, Euclid-like algorithm: with Carl Jacobi's 1837
extension to composite moduli, it can be evaluated without factoring
anything. Alberto Tonelli's 1891 algorithm then finds the square root
itself.

.. math::

   \left(\frac{p}{q}\right)\left(\frac{q}{p}\right)
   = (-1)^{\frac{p-1}{2}\cdot\frac{q-1}{2}}

*Implementation:* :func:`mathematicskit.number_theory.systems.quadratic_residues.legendre_symbol`
evaluates the symbol by Euler's criterion
:math:`a^{(p-1)/2} \bmod p`.
:func:`~mathematicskit.number_theory.systems.quadratic_residues.jacobi_symbol`
uses reciprocity to evaluate the Jacobi symbol without factoring, and
:func:`~mathematicskit.number_theory.systems.quadratic_residues.sqrt_mod`
computes modular square roots with the Tonelli-Shanks algorithm. The
test suite checks the reciprocity law for every pair of odd primes
below 200.

*References:* C. F. Gauss, *Disquisitiones Arithmeticae* (Leipzig:
Fleischer, 1801), Section IV, Articles 94-152; A.-M. Legendre,
"Recherches d'analyse indéterminée," Histoire de l'Académie Royale des
Sciences (1785), 465-559; A. Tonelli, "Bemerkung über die Auflösung
quadratischer Congruenzen," Nachrichten von der Königlichen
Gesellschaft der Wissenschaften zu Göttingen (1891), 344-346.

.. minigallery:: ../../examples/number_theory/quadratic_residues/plot_01_quadratic_reciprocity.py

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

1837 -- Dirichlet's Theorem on Primes in Arithmetic Progressions
----------------------------------------------------------------

Euclid's argument shows there are infinitely many primes, but does the
progression :math:`1, 11, 21, 31, \dots` contain infinitely many?
Peter Gustav Lejeune Dirichlet proved in 1837 that every progression
:math:`a, a+q, a+2q, \dots` with :math:`\gcd(a, q) = 1` does. His proof
generalized Euler's product formula (above), replacing :math:`\zeta(s)`
with what are now called Dirichlet :math:`L`-functions, built from the
characters of the group :math:`(\mathbb{Z}/q\mathbb{Z})^{\times}`. This
was the first proof of a result about integers that relied on complex
analysis and group characters, and it launched analytic number theory
as a subject. The coprime residue classes eventually share the primes
equally, but not uniformly along the way. Pafnuty Chebyshev noticed in
1853 that primes :math:`\equiv 3 \pmod 4` almost always outnumber those
:math:`\equiv 1 \pmod 4`, and John Leech found in 1957 that the lead
first changes hands at 26,861. Michael Rubinstein and Peter Sarnak
explained this "Chebyshev's bias" in 1994.

.. math::

   \pi(x; q, a) \sim \frac{\pi(x)}{\varphi(q)} \qquad (\gcd(a, q) = 1)

*Implementation:* :func:`mathematicskit.number_theory.systems.prime_distribution.primes_in_progression`
lists the primes in a residue class up to a bound. The gallery example
uses it to show the equal split of primes modulo 10 and the mod-4 prime
race.

*References:* P. G. L. Dirichlet, "Beweis des Satzes, dass jede
unbegrenzte arithmetische Progression, deren erstes Glied und Differenz
ganze Zahlen ohne gemeinschaftlichen Factor sind, unendlich viele
Primzahlen enthält," Abhandlungen der Königlich Preussischen Akademie
der Wissenschaften zu Berlin (1837), 45-81; M. Rubinstein and P. Sarnak,
"Chebyshev's Bias," Experimental Mathematics 3(3) (1994), 173-197.

.. minigallery:: ../../examples/number_theory/prime_distribution/plot_02_dirichlet_prime_races.py

1876-1930 -- Lucas, Lehmer, and Mersenne Primes
-----------------------------------------------

Numbers of the form :math:`M_p = 2^p - 1` have attracted attention
since Euclid, who showed that each prime :math:`M_p` gives a perfect
number :math:`2^{p-1}M_p`. They take their name from Marin Mersenne,
whose 1644 list of exponents was partly wrong. In 1876 Édouard Lucas
proved by hand that :math:`M_{127}`, a 39-digit number, is prime. It
remained the largest known prime for 75 years. His method, completed
by Derrick Henry Lehmer in 1930, needs only :math:`p - 2` squarings
modulo :math:`M_p`, far fewer than any general primality test. Raphael
Robinson ran it on the SWAC computer in 1952 and found five new
Mersenne primes, the first found by a computer. The Great Internet
Mersenne Prime Search still relies on the same test, which is why
nearly every record prime since then has been a Mersenne prime.

.. math::

   s_0 = 4, \quad s_{k+1} = s_k^2 - 2 \bmod M_p; \qquad
   M_p \ \text{prime} \iff s_{p-2} \equiv 0 \pmod{M_p}

*Implementation:* :func:`mathematicskit.number_theory.systems.primality.lucas_lehmer`
runs the test with Python's arbitrary-precision integers. The test
suite recovers every Mersenne exponent below 650 and cross-checks each
result against
:func:`~mathematicskit.number_theory.systems.primality.is_prime_miller_rabin`.

*References:* É. Lucas, "Théorie des fonctions numériques simplement
périodiques," American Journal of Mathematics 1 (1878), 184-240 and
289-321; D. H. Lehmer, "An Extended Theory of Lucas' Functions," Annals
of Mathematics 31(3) (1930), 419-448.

.. minigallery:: ../../examples/number_theory/primality/plot_02_lucas_lehmer.py

1896 -- The Prime Number Theorem
--------------------------------

Around 1792, the teenage Carl Friedrich Gauss studied tables of primes
and noticed that near :math:`x` their density is about
:math:`1/\ln x`. That suggests counting the primes up to :math:`x` with
the logarithmic integral :math:`\operatorname{li}(x)`. Adrien-Marie
Legendre published a similar guess in 1798. Pafnuty Chebyshev showed
in 1850 that :math:`\pi(x)` stays within about 11% of
:math:`x/\ln x`, and Bernhard Riemann's 1859 memoir tied the exact
error to the zeros of the zeta function (above). In 1896 Jacques
Hadamard and Charles-Jean de la Vallée Poussin independently proved the
conjecture by showing that :math:`\zeta(s)` has no zeros on the line
:math:`\operatorname{Re}(s) = 1`. Atle Selberg and Paul Erdős found an
"elementary" proof, with no complex analysis, in 1949.

.. math::

   \pi(x) \sim \operatorname{li}(x) = \int_0^x \frac{dt}{\ln t}
   \sim \frac{x}{\ln x}

*Implementation:* :func:`mathematicskit.number_theory.systems.prime_distribution.prime_counting`
evaluates :math:`\pi(x)` for a whole array of :math:`x` from a single
sieve, and
:func:`~mathematicskit.number_theory.systems.prime_distribution.logarithmic_integral`
evaluates :math:`\operatorname{li}(x) = \operatorname{Ei}(\ln x)` with
:func:`scipy.special.expi`. At :math:`x = 10^7` the logarithmic
integral is off by 0.05%, :math:`x/\ln x` by 7%.

*References:* J. Hadamard, "Sur la distribution des zéros de la fonction
:math:`\zeta(s)` et ses conséquences arithmétiques," Bulletin de la
Société Mathématique de France 24 (1896), 199-220; C.-J. de la Vallée
Poussin, "Recherches analytiques sur la théorie des nombres premiers,"
Annales de la Société Scientifique de Bruxelles 20 (1896), 183-256;
B. Riemann, "Ueber die Anzahl der Primzahlen unter einer gegebenen
Grösse," Monatsberichte der Königlichen Preussischen Akademie der
Wissenschaften zu Berlin (1859), 671-680.

.. minigallery:: ../../examples/number_theory/prime_distribution/plot_01_prime_number_theorem.py

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

1975 -- Pollard's Rho Factorization
-----------------------------------

RSA's security (above) rests on factoring being hard, which makes
faster factoring algorithms a matter of practical importance. John
Pollard's 1975 "Monte Carlo method" iterates the pseudo-random map
:math:`x \mapsto x^2 + c \bmod n`. Modulo an unknown prime factor
:math:`p` of :math:`n`, the sequence must repeat within about
:math:`\sqrt{p}` steps (the birthday paradox), and its path traces
the Greek letter :math:`\rho`. Robert Floyd's tortoise-and-hare trick
detects the repeat, and a gcd with :math:`n` then reveals :math:`p`.
The method finds a factor :math:`p` in about :math:`\sqrt p` steps
using almost no memory, compared with :math:`p` steps for trial
division. In 1980 Richard Brent and Pollard used a refined version to
factor the eighth Fermat number :math:`2^{256}+1`.

.. math::

   x_{i+1} = x_i^2 + c \bmod n, \qquad
   d = \gcd(|x_i - x_{2i}|, n)

*Implementation:* :func:`mathematicskit.number_theory.systems.factorization.pollard_rho`
implements the method with Floyd cycle detection, restarting with a new
constant :math:`c` if the cycle closes modulo every factor at once. It
returns a :class:`~mathematicskit.number_theory.core.base.PollardRhoResult`
with the factor, cofactor, and step count. The gallery example
factors the Fermat number :math:`2^{64}+1 = 274177 \times
67280421310721` (first factored by Fortuné Landry in 1880) and shows
the step count growing like :math:`\sqrt p`.

*References:* J. M. Pollard, "A Monte Carlo Method for Factorization,"
BIT Numerical Mathematics 15(3) (1975), 331-334; R. P. Brent and J. M.
Pollard, "Factorization of the Eighth Fermat Number," Mathematics of
Computation 36(154) (1981), 627-630.

.. minigallery:: ../../examples/number_theory/factorization/plot_01_pollard_rho.py

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
