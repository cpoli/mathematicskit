Breakthroughs in Combinatorics
=================================


.. include:: /_generated/nav/combinatorics.rst

.. epigraph::

   "Combinatorics is the slums of topology." -- a wry (and much disputed)
   remark long attributed, probably apocryphally, to Gian-Carlo Rota

Counting things exactly -- not approximately, not asymptotically, but
exactly -- turns out to be a surprisingly deep subject, full of hidden
bijections between problems that look nothing alike on the surface. This
chronology traces the ideas behind :mod:`mathkit.combinatorics`, from
Pascal's triangle to the partition function whose exact evaluation
Ramanujan and Hardy needed an entirely new method of complex analysis to
reach.

.. contents:: Timeline
   :local:
   :depth: 1

1665 -- Pascal's Triangle (and Its Much Older Ancestors)
----------------------------------------------------------------

Blaise Pascal's 1654 treatise *Traite du triangle arithmetique*
(published 1665) gave the triangular array of binomial coefficients its
Western name and a thorough combinatorial treatment, but the triangle
itself had already been independently discovered centuries earlier: in
10th-century India (Halayudha's commentary on Pingala), 11th-century
Persia (Al-Karaji and Omar Khayyam), and 13th-century China (Jia Xian,
popularized by Yang Hui, whose name the triangle carries in Chinese
mathematics today). Every version rests on the same recurrence: each
entry is the sum of the two entries above it.

.. math::

   \binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}

*Implementation:* :func:`mathkit.combinatorics.systems.pascals_triangle.pascals_triangle`
builds exactly this triangle via the recurrence, kept hand-rolled purely
as a pedagogical illustration alongside the primary API,
:func:`~mathkit.combinatorics.systems.counting.combinations_count`,
which wraps :func:`scipy.special.comb`.

*References:* B. Pascal, *Traite du triangle arithmetique* (Paris:
Desprez, 1665).

.. minigallery:: ../../examples/combinatorics/pascals_triangle/plot_01_triangle_and_binomials.py

1713 -- Bernoulli, Montmort, and the Derangement Problem
------------------------------------------------------------------

Pierre Remond de Montmort's 1708 *Essai d'analyse sur les jeux de
hasard* posed the "probleme des rencontres": shuffle a deck and ask how
likely no card lands in its original position -- a permutation with no
fixed point, later named a "derangement." Montmort and Nicolaus
Bernoulli, corresponding between 1710 and 1713, worked out the exact
count via what is now recognized as an early, concrete instance of the
inclusion-exclusion principle -- alternately over- and under-counting
permutations that fix at least one, at least two, at least three points,
and so on.

.. math::

   D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!}

*Implementation:* :func:`mathkit.combinatorics.systems.inclusion_exclusion.derangement_count`
implements exactly this count via the equivalent integer recurrence
:math:`D_n = (n-1)(D_{n-1}+D_{n-2})`, and
:func:`~mathkit.combinatorics.systems.inclusion_exclusion.union_size_inclusion_exclusion`
implements the general inclusion-exclusion principle the derangement
count is a specific application of.

*References:* P. R. de Montmort, *Essai d'analyse sur les jeux de
hasard*, 2nd ed. (Paris: Quillau, 1713).

.. minigallery:: ../../examples/combinatorics/inclusion_exclusion/plot_01_hat_check_problem.py

1751 -- Euler and the Catalan Numbers
-------------------------------------------

Leonhard Euler's 1751 correspondence with Christian Goldbach posed and
solved the problem of counting the ways to dissect a convex polygon into
triangles using non-crossing diagonals -- the first appearance of the
sequence :math:`1, 1, 2, 5, 14, 42, \dots` now named for Eugene Charles
Catalan, whose own 1838 paper on a closely related bracketing problem
gave the sequence its modern name and recurrence. The same numbers count
an enormous variety of unrelated-looking combinatorial objects: balanced
parenthesizations, binary trees, and non-crossing lattice paths among
them, one of the most striking examples in mathematics of a single
integer sequence with many independent combinatorial meanings.

.. math::

   C_0 = 1, \qquad C_{n+1} = \sum_{i=0}^{n} C_i C_{n-i}

*Implementation:* :func:`mathkit.combinatorics.systems.special_numbers.catalan_number`
implements exactly Catalan's own recurrence (rather than the equivalent
closed form :math:`\binom{2n}{n}/(n+1)`, kept as a cross-check in this
domain's tests), and is verified directly against brute-force
enumeration of balanced-parenthesis strings.

*References:* E. Catalan, "Note sur une equation aux differences
finies," Journal de Mathematiques Pures et Appliquees 3 (1838), 508-516.

.. minigallery:: ../../examples/combinatorics/special_numbers/plot_01_catalan_and_stirling.py

1730 -- Stirling's Numbers
--------------------------------

James Stirling's 1730 treatise *Methodus Differentialis* introduced two
families of numbers converting between ordinary powers and falling
factorials -- the Stirling numbers of the first kind (counting
permutations by their number of cycles) and second kind (counting the
ways to partition a labeled set into a fixed number of non-empty,
unlabeled blocks), the combinatorial machinery underlying the Bell
numbers (the *total* number of ways to partition a set into any number
of blocks at all), named for Eric Temple Bell's 1934 paper studying
their generating function.

*Implementation:* :func:`mathkit.combinatorics.systems.special_numbers.stirling_first_kind`
and :func:`~mathkit.combinatorics.systems.special_numbers.stirling_second_kind`
implement both families via their defining recurrences;
:func:`mathkit.combinatorics.utils.bell_number.bell_number` sums the
second kind over every possible number of blocks.

*References:* J. Stirling, *Methodus Differentialis* (London: Bowyer,
1730).

.. minigallery:: ../../examples/combinatorics/special_numbers/plot_01_catalan_and_stirling.py

1674 -- 1918 -- Young Diagrams and Integer Partitions
------------------------------------------------------------

The partition function :math:`p(n)`, counting the ways to write
:math:`n` as a sum of positive integers regardless of order, was studied
combinatorially at least as far back as Leibniz's 1674 correspondence,
and Euler developed its generating function
:math:`\prod_{k\geq1}(1-x^k)^{-1}` in the mid-18th century. Alfred
Young's 1900-1902 diagrams -- left-justified rows of boxes, one row per
part, in non-increasing length -- gave partitions their standard visual
representation, whose *conjugate* (reflecting the diagram across its
main diagonal, swapping rows and columns) reveals a beautiful hidden
symmetry of the partition function. Srinivasa Ramanujan and G. H.
Hardy's 1918 circle method finally gave :math:`p(n)` a genuine
asymptotic formula, after nearly two and a half centuries of the
function resisting any closed form at all.

*Implementation:* :func:`mathkit.combinatorics.systems.partitions.partition_function`
computes :math:`p(n)` exactly via Euler's own pentagonal-number-theorem
recurrence (rather than Hardy and Ramanujan's asymptotic circle-method
formula, which this domain does not implement);
:func:`~mathkit.combinatorics.systems.partitions.integer_partitions`
enumerates every partition explicitly, and
:class:`mathkit.combinatorics.core.base.YoungDiagram` implements exactly
Young's diagram and its conjugation.

*References:* A. Young, "On Quantitative Substitutional Analysis,"
Proceedings of the London Mathematical Society 33(1) (1900), 97-146; G.
H. Hardy and S. Ramanujan, "Asymptotic Formulae in Combinatory
Analysis," Proceedings of the London Mathematical Society 17(1) (1918),
75-115.

.. minigallery:: ../../examples/combinatorics/partitions/plot_01_partitions_and_young_diagrams.py

See Also
--------

- :doc:`/api/combinatorics`
- :doc:`/history/number_theory_breakthroughs`
- :doc:`/history/probability_breakthroughs`
