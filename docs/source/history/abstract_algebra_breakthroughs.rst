Breakthroughs in Abstract Algebra
====================================


.. include:: /_generated/nav/abstract_algebra.rst

.. epigraph::

   "Algebra is generous; she often gives more than is asked of her." --
   Jean le Rond d'Alembert, as widely quoted in the mathematical
   folklore surrounding him

Abstract algebra's founding insight is that the *same* structural
skeleton -- a set with an operation satisfying a short list of axioms --
underlies symmetry groups, number systems, and equation-solving alike,
once the specific objects are stripped away and only the relations
between them are kept. This chronology traces the ideas behind
:mod:`mathkit.abstract_algebra`, from a young mathematician's
night-before-a-duel manuscript to the finite fields that now protect
every error-corrected digital signal.

.. contents:: Timeline
   :local:
   :depth: 1

1832 -- Galois and the Birth of Group Theory
--------------------------------------------------

Evariste Galois, in a letter written the night before the duel that
would kill him at twenty, sketched a theory relating the solvability of
a polynomial equation by radicals to the structure of a group of
permutations of its roots -- introducing, along the way, the word
"group" (*groupe*) itself in essentially its modern technical sense.
Galois's own manuscripts were unpublished and largely unread until
Joseph Liouville edited and published them in 1846, fourteen years after
Galois's death; only then did the mathematical community grasp that
group theory could resolve a two-and-a-half-century-old open problem
(why the general quintic has no radical solution) by translating it into
a purely structural question about permutations.

*Implementation:* :class:`mathkit.abstract_algebra.systems.groups.PermutationGroup`
implements exactly Galois's permutation groups, building the full
element set by closing a set of generators under composition -- the
symmetric group :math:`S_3` example in this domain's tests is
non-abelian for precisely the structural reason Galois's theory turns
on.

*References:* E. Galois, "Manuscrits de Evariste Galois," ed. J.
Liouville, Journal de Mathematiques Pures et Appliquees 11 (1846),
381-444.

.. minigallery:: ../../examples/abstract_algebra/groups/plot_01_cyclic_vs_symmetric.py

1854 -- Cayley's Abstract Definition and the Cayley Table
-----------------------------------------------------------------

Arthur Cayley's 1854 paper "On the Theory of Groups, as Depending on the
Symbolic Equation :math:`\theta^n=1`" gave the first fully abstract
definition of a group -- a set with a binary operation satisfying
closure, associativity, an identity, and inverses -- independent of any
particular representation as permutations or numbers, and introduced the
square multiplication table (now named for him) as the concrete way to
display any small finite group's complete structure at a glance.

*Implementation:* :meth:`mathkit.abstract_algebra.core.base.FiniteGroup.cayley_table`
builds exactly this table for any concrete group implementation,
purely from the abstract ``elements``/``operate`` interface Cayley's own
1854 definition specifies.

*References:* A. Cayley, "On the Theory of Groups, as Depending on the
Symbolic Equation :math:`\theta^n = 1`," Philosophical Magazine Series 4,
7(42) (1854), 40-47.

.. minigallery:: ../../examples/abstract_algebra/groups/plot_01_cyclic_vs_symmetric.py

1770 -- 1878 -- Lagrange's Theorem
----------------------------------------

Joseph-Louis Lagrange's 1770-71 memoir on the solvability of equations
implicitly used, without stating it as a general theorem about abstract
groups (which did not yet exist as a concept), the fact that a
subgroup's size must divide the size of the group containing it. Camille
Jordan's 1870 treatise gave the result its first fully general, modern
statement and proof: a finite group's subgroup order always divides the
group's own order, with the group splitting cleanly into disjoint cosets
of the subgroup, each exactly the subgroup's size.

.. math::

   |H| \text{ divides } |G|, \qquad |G| = |H| \cdot [G:H]

*Implementation:* :func:`mathkit.abstract_algebra.systems.subgroups.cyclic_subgroup`
and :func:`~mathkit.abstract_algebra.systems.subgroups.all_subgroups`
find a group's subgroups directly;
:func:`~mathkit.abstract_algebra.systems.subgroups.left_cosets` computes
exactly the coset partition Lagrange's theorem describes, verified
directly in this domain's tests to always divide evenly.

*References:* C. Jordan, *Traite des substitutions et des equations
algebriques* (Paris: Gauthier-Villars, 1870).

.. minigallery:: ../../examples/abstract_algebra/subgroups/plot_01_lagranges_theorem.py

1830 -- 1893 -- Galois Fields
------------------------------------

Evariste Galois's 1830 paper "Sur la theorie des nombres" extended
modular arithmetic beyond prime moduli :math:`p` to prime-power moduli
:math:`p^n`, constructing what are now called Galois fields
:math:`\mathrm{GF}(p^n)` in his honor: the integers mod :math:`p`
extended by adjoining a root of an irreducible polynomial of degree
:math:`n`, exactly as the complex numbers extend the reals by adjoining
a root of :math:`x^2+1`. E. H. Moore's 1893 address to the International
Mathematical Congress gave the theory of finite fields its first fully
rigorous, general modern treatment, proving that every finite field has
prime-power order and that fields of the same order are all isomorphic.

*Implementation:* :class:`mathkit.abstract_algebra.systems.finite_fields.GF`
implements exactly this construction:
:func:`~mathkit.abstract_algebra.systems.finite_fields.find_irreducible_polynomial`
finds a defining irreducible polynomial by brute-force search, and field
arithmetic reduces every product modulo it -- the same construction that
underlies Reed-Solomon error-correcting codes and the Advanced
Encryption Standard's byte-level arithmetic in :math:`\mathrm{GF}(2^8)`.

*References:* E. Galois, "Sur la theorie des nombres," Bulletin des
Sciences Mathematiques 13 (1830), 428-435; E. H. Moore, "A Doubly-Infinite
System of Simple Groups," Bulletin of the New York Mathematical Society
3(3) (1893), 73-78.

.. minigallery:: ../../examples/abstract_algebra/finite_fields/plot_01_gf8_multiplication_table.py

1801 -- Polynomial Rings and Euclidean Division
-------------------------------------------------------

Gauss's *Disquisitiones Arithmeticae* (see the number-theory chronology)
observed that polynomials with coefficients in a field support the same
division-with-remainder structure as ordinary integers -- given
polynomials :math:`p` and :math:`q\neq0`, there is a unique quotient and
remainder with the remainder's degree strictly less than :math:`q`'s --
making the ring of polynomials over a field a Euclidean domain in
exactly the sense integers are, with its own Euclidean algorithm for
computing a greatest common divisor of two polynomials.

*Implementation:* :class:`mathkit.abstract_algebra.core.base.Polynomial`
implements exactly this division algorithm (over :math:`\mathbb{Q}` or a
finite field), and :func:`mathkit.abstract_algebra.systems.polynomial_ring.poly_gcd`
implements the resulting polynomial Euclidean algorithm, structurally
identical to :func:`mathkit.number_theory.systems.modular_arithmetic.extended_gcd`'s
integer version.

.. minigallery:: ../../examples/abstract_algebra/polynomial_ring/plot_01_division_and_gcd.py

See Also
--------

- :doc:`/api/abstract_algebra`
- :doc:`/history/number_theory_breakthroughs`
- :doc:`/history/combinatorics_breakthroughs`
