Breakthroughs in Combinatorics
==============================


.. include:: /_generated/nav/combinatorics.rst

.. epigraph::

   "Combinatorics is the pegboard of mathematics."
   -- Gian-Carlo Rota

Counting things exactly -- not approximately, not asymptotically, but
exactly -- turns out to be a surprisingly deep subject, full of hidden
bijections between problems that look nothing alike. This chronology
traces the ideas behind :mod:`mathematicskit.combinatorics`, from
Pascal's triangle to the partition function, whose growth Srinivasa
Ramanujan and Godfrey Harold Hardy could only capture by inventing a
new method of complex analysis.

.. contents:: Timeline
   :local:
   :depth: 1

1654-1665 -- Pascal's Triangle (and Its Much Older Ancestors)
-------------------------------------------------------------

Blaise Pascal wrote his *Traité du triangle arithmétique* in 1654, and
it was published posthumously in 1665. It gave the triangular array of
binomial coefficients its Western name and a thorough combinatorial
treatment. The triangle itself had been discovered independently
centuries earlier: in 10th-century India (Halayudha's commentary on
Pingala), in Persia around 1000 (Al-Karaji, and later Omar Khayyam),
and in 11th-century China (Jia Xian). Yang Hui popularized it in China
in 1261, and it carries his name in Chinese mathematics today. Every
version rests on the same recurrence: each entry is the sum of the two
entries above it.

.. math::

   \binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}

*Implementation:* :func:`mathematicskit.combinatorics.systems.pascals_triangle.pascals_triangle`
builds the triangle from this recurrence. It is hand-rolled purely as a
teaching illustration; the primary API is
:func:`~mathematicskit.combinatorics.systems.counting.combinations_count`,
which wraps :func:`scipy.special.comb`.

*References:* B. Pascal, *Traité du triangle arithmétique* (Paris:
Guillaume Desprez, 1665).

.. minigallery:: ../../examples/combinatorics/pascals_triangle/plot_01_triangle_and_binomials.py

1674-1918 -- Young Diagrams and Integer Partitions
--------------------------------------------------

The partition function :math:`p(n)` counts the ways to write :math:`n`
as a sum of positive integers, ignoring order. Gottfried Wilhelm Leibniz
studied it in unpublished notes from 1674, and Leonhard Euler found its
generating function :math:`\prod_{k\geq1}(1-x^k)^{-1}` in the 1740s.
Alfred Young's diagrams, introduced in 1900, draw a partition as
left-justified rows of boxes, one row per part, in non-increasing
length. Reflecting a diagram across its main diagonal gives the
*conjugate* partition, which reveals hidden symmetries: for example,
the number of partitions of :math:`n` into :math:`k` parts equals the
number whose largest part is :math:`k`. No closed form for :math:`p(n)`
was ever found, but in 1918 Godfrey Harold Hardy and Srinivasa Ramanujan used
their new circle method to give it a precise asymptotic formula.

*Implementation:* :func:`mathematicskit.combinatorics.systems.partitions.partition_function`
computes :math:`p(n)` exactly with Euler's pentagonal-number-theorem
recurrence. This domain does not implement the Hardy-Ramanujan
asymptotic formula.
:func:`~mathematicskit.combinatorics.systems.partitions.integer_partitions`
lists every partition explicitly, and
:class:`mathematicskit.combinatorics.core.base.YoungDiagram` implements
Young's diagram and its conjugate.

*References:* A. Young, "On Quantitative Substitutional Analysis,"
Proceedings of the London Mathematical Society 33(1) (1900), 97-146; G.
H. Hardy and S. Ramanujan, "Asymptotic Formulae in Combinatory
Analysis," Proceedings of the London Mathematical Society 17(1) (1918),
75-115.

.. minigallery:: ../../examples/combinatorics/partitions/plot_01_partitions_and_young_diagrams.py

1708-1713 -- Bernoulli, Montmort, and the Derangement Problem
-------------------------------------------------------------

Pierre Rémond de Montmort's 1708 *Essay d'analyse sur les jeux de
hazard* posed the *problème des rencontres*: shuffle a deck, and ask
how likely it is that no card ends up in its original position. Such a
permutation, with no fixed point, is now called a "derangement."
Montmort and Nicolaus Bernoulli, corresponding between 1710 and 1713,
worked out the exact count. Their method is an early instance of the
inclusion-exclusion principle: alternately subtract and add the
permutations that fix at least one, at least two, at least three
points, and so on.

.. math::

   D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!}

*Implementation:* :func:`mathematicskit.combinatorics.systems.inclusion_exclusion.derangement_count`
computes this count with the equivalent integer recurrence
:math:`D_n = (n-1)(D_{n-1}+D_{n-2})`.
:func:`~mathematicskit.combinatorics.systems.inclusion_exclusion.union_size_inclusion_exclusion`
implements the general inclusion-exclusion principle, of which the
derangement count is one application.

*References:* P. R. de Montmort, *Essay d'analyse sur les jeux de
hazard*, 2nd ed. (Paris: Jacque Quillau, 1713).

.. minigallery:: ../../examples/combinatorics/inclusion_exclusion/plot_01_hat_check_problem.py

1730 -- Stirling's Numbers
--------------------------

James Stirling's 1730 treatise *Methodus Differentialis* introduced two
families of numbers that convert between ordinary powers and falling
factorials. In modern terms, Stirling numbers of the first kind count
permutations by their number of cycles. Stirling numbers of the second
kind count the ways to partition a labeled set into a fixed number of
non-empty, unlabeled blocks. Summing the second kind over every
possible number of blocks gives the Bell numbers, the *total* number of
ways to partition a set. They are named for Eric Temple Bell, whose
1934 papers studied them.

*Implementation:* :func:`mathematicskit.combinatorics.systems.special_numbers.stirling_first_kind`
and :func:`~mathematicskit.combinatorics.systems.special_numbers.stirling_second_kind`
implement both families with their defining recurrences;
:func:`mathematicskit.combinatorics.utils.bell_number.bell_number` sums the
second kind over every possible number of blocks.

*References:* J. Stirling, *Methodus Differentialis* (London: Bowyer,
1730).

.. minigallery:: ../../examples/combinatorics/special_numbers/plot_01_catalan_and_stirling.py

1751-1838 -- Euler, Catalan, and the Catalan Numbers
----------------------------------------------------

In a 1751 letter to Christian Goldbach, Leonhard Euler posed and solved
the problem of counting the ways to cut a convex polygon into triangles
with non-crossing diagonals. This was the first appearance of the
sequence :math:`1, 1, 2, 5, 14, 42, \dots`. Eugène Charles Catalan's
1838 paper connected the same numbers to the problem of bracketing a
product, and the sequence was later named after him. The same numbers
count a remarkable variety of unrelated-looking objects, among them
balanced parenthesizations, binary trees, and non-crossing lattice
paths. Few integer sequences have as many independent combinatorial
meanings.

.. math::

   C_0 = 1, \qquad C_{n+1} = \sum_{i=0}^{n} C_i C_{n-i}

*Implementation:* :func:`mathematicskit.combinatorics.systems.special_numbers.catalan_number`
implements this recurrence, and is checked against brute-force
enumeration of balanced-parenthesis strings. This domain's tests keep
the equivalent closed form :math:`\binom{2n}{n}/(n+1)` as a
cross-check.

*References:* E. Catalan, "Note sur une équation aux différences
finies," Journal de Mathématiques Pures et Appliquées 3 (1838), 508-516.

.. minigallery:: ../../examples/combinatorics/special_numbers/plot_01_catalan_and_stirling.py

See Also
--------

- :doc:`/api/combinatorics`
- :doc:`/history/number_theory_breakthroughs`
- :doc:`/history/probability_breakthroughs`
