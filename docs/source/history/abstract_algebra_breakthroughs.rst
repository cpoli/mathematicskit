Breakthroughs in Abstract Algebra
=================================


.. include:: /_generated/nav/abstract_algebra.rst

.. epigraph::

   "Algebra is generous; she often gives more than is asked of her."
   -- attributed to Jean le Rond d'Alembert

Abstract algebra rests on one insight: the *same* structural skeleton --
a set with an operation that satisfies a short list of axioms -- lies
behind symmetry groups, number systems, and equation-solving alike. Once
the specific objects are stripped away, only the relations between them
remain. This chronology traces the ideas behind
:mod:`mathematicskit.abstract_algebra`, from Renaissance polynomial
division and Évariste Galois's letter written the night before his fatal
duel to the finite fields that now protect every error-corrected digital
signal.

.. contents:: Timeline
   :local:
   :depth: 1

1585-1801 -- Polynomial Rings and Euclidean Division
----------------------------------------------------

Polynomials with coefficients in a field can be divided with remainder
exactly as integers can: given polynomials :math:`p` and :math:`q\neq0`,
there is a unique quotient and a unique remainder whose degree is
strictly less than that of :math:`q`. Simon Stevin's 1585
*L'Arithmétique* used this division to run Euclid's algorithm on
polynomials, computing their greatest common divisor. Carl Friedrich
Gauss's 1801 *Disquisitiones Arithmeticae* (see the number-theory
chronology) then treated polynomial congruences with the same methods
it applied to integers. In modern language, the ring of polynomials over
a field is a Euclidean domain, just as the integers are.

*Implementation:* :class:`mathematicskit.abstract_algebra.core.base.Polynomial`
implements this division algorithm (over :math:`\mathbb{Q}` or a
finite field), and :func:`mathematicskit.abstract_algebra.systems.polynomial_ring.poly_gcd`
implements the resulting polynomial Euclidean algorithm. It has the same
structure as the integer version,
:func:`mathematicskit.number_theory.systems.modular_arithmetic.extended_gcd`.

*References:* S. Stevin, *L'Arithmétique* (Leiden: Plantin, 1585); C. F.
Gauss, *Disquisitiones Arithmeticae* (Leipzig: Fleischer, 1801).

.. minigallery:: ../../examples/abstract_algebra/polynomial_ring/plot_01_division_and_gcd.py

1770-1870 -- Lagrange's Theorem
-------------------------------

Joseph-Louis Lagrange's 1770-71 memoir on the algebraic solution of
equations used a special case of the theorem now named for him: the
size of a subgroup divides the size of the whole group. Lagrange stated
it for permutations of an equation's roots, because abstract groups did
not yet exist as a concept. Camille Jordan's 1870 treatise gave the
result a general statement and proof. A finite group splits into
disjoint cosets of any subgroup, each exactly the subgroup's size, so
the subgroup's order must divide the group's order.

.. math::

   |H| \text{ divides } |G|, \qquad |G| = |H| \cdot [G:H]

*Implementation:* :func:`mathematicskit.abstract_algebra.systems.subgroups.cyclic_subgroup`
and :func:`~mathematicskit.abstract_algebra.systems.subgroups.all_subgroups`
find a group's subgroups directly.
:func:`~mathematicskit.abstract_algebra.systems.subgroups.left_cosets`
computes the coset partition that Lagrange's theorem describes, and this
domain's tests check that it always divides the group evenly.

*References:* J.-L. Lagrange, "Réflexions sur la résolution algébrique
des équations," Nouveaux Mémoires de l'Académie Royale des Sciences et
Belles-Lettres de Berlin (1770-1771); C. Jordan, *Traité des
substitutions et des équations algébriques* (Paris: Gauthier-Villars,
1870).

.. minigallery:: ../../examples/abstract_algebra/subgroups/plot_01_lagranges_theorem.py

1830-1893 -- Galois Fields
--------------------------

Évariste Galois's 1830 paper "Sur la théorie des nombres" extended
modular arithmetic from a prime modulus :math:`p` to fields with
:math:`p^n` elements, now called Galois fields :math:`\mathrm{GF}(p^n)`
in his honor. The construction takes the integers mod :math:`p` and
adjoins a root of an irreducible polynomial of degree :math:`n`, just as
the complex numbers extend the reals by adjoining a root of
:math:`x^2+1`. Eliakim Hastings Moore's paper at the 1893 International
Mathematical Congress in Chicago gave finite fields their first
rigorous general treatment. Moore proved that every finite field has
prime-power order and that any two fields of the same order are
isomorphic.

*Implementation:* :class:`mathematicskit.abstract_algebra.systems.finite_fields.GF`
implements this construction.
:func:`~mathematicskit.abstract_algebra.systems.finite_fields.find_irreducible_polynomial`
finds a defining irreducible polynomial by brute-force search, and the
field arithmetic reduces every product modulo that polynomial. The same
construction underlies Reed-Solomon error-correcting codes and the
byte-level arithmetic of the Advanced Encryption Standard in
:math:`\mathrm{GF}(2^8)`.

*References:* É. Galois, "Sur la théorie des nombres," Bulletin des
Sciences Mathématiques 13 (1830), 428-435; E. H. Moore, "A
Doubly-Infinite System of Simple Groups," Bulletin of the New York
Mathematical Society 3(3) (1893), 73-78.

.. minigallery:: ../../examples/abstract_algebra/finite_fields/plot_01_gf8_multiplication_table.py

1832-1846 -- Galois and the Birth of Group Theory
-------------------------------------------------

In a letter written the night before the duel that killed him at twenty,
Évariste Galois summarized a theory linking whether a polynomial
equation can be solved by radicals to the structure of a group of
permutations of its roots. Along the way he introduced the word "group"
(*groupe*) in essentially its modern technical sense. Niels Henrik Abel
had already proved in 1824 that the general quintic has no solution by
radicals; Galois's theory went further and decided, for any given
equation, whether it has one. His manuscripts stayed unpublished and
largely unread until Joseph Liouville edited and published them in
1846, fourteen years after Galois's death. Only then did mathematicians
see that a question about equations could be turned into a purely
structural question about permutations.

*Implementation:* :class:`mathematicskit.abstract_algebra.systems.groups.PermutationGroup`
implements Galois's permutation groups, building the full element set
by closing a set of generators under composition. The symmetric group
:math:`S_3` in this domain's tests is non-abelian, and non-commutativity
is exactly the structural feature Galois's theory turns on.

*References:* É. Galois, "Œuvres mathématiques d'Évariste Galois," ed.
J. Liouville, Journal de Mathématiques Pures et Appliquées 11 (1846),
381-444.

.. minigallery:: ../../examples/abstract_algebra/groups/plot_01_cyclic_vs_symmetric.py

1854 -- Cayley's Abstract Definition and the Cayley Table
---------------------------------------------------------

Arthur Cayley's 1854 paper "On the Theory of Groups, as Depending on the
Symbolic Equation :math:`\theta^n=1`" gave the first abstract definition
of a group: a set with a binary operation, independent of any
particular representation as permutations or numbers. The same paper
introduced the square multiplication table, now named for him, which
displays the complete structure of a small finite group at a glance.

*Implementation:* :meth:`mathematicskit.abstract_algebra.core.base.FiniteGroup.cayley_table`
builds this table for any concrete group implementation, using only the
abstract ``elements``/``operate`` interface, in the spirit of Cayley's
definition.

*References:* A. Cayley, "On the Theory of Groups, as Depending on the
Symbolic Equation :math:`\theta^n = 1`," Philosophical Magazine Series 4,
7(42) (1854), 40-47.

.. minigallery:: ../../examples/abstract_algebra/groups/plot_01_cyclic_vs_symmetric.py

See Also
--------

- :doc:`/api/abstract_algebra`
- :doc:`/history/number_theory_breakthroughs`
- :doc:`/history/combinatorics_breakthroughs`
