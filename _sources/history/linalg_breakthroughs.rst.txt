Breakthroughs in Numerical Linear Algebra
============================================


.. include:: /_generated/nav/linalg.rst

.. epigraph::

   "Linear algebra is the ultimate service subject." -- Gilbert Strang,
   as quoted widely in his own lecture courses

Solving :math:`Ax=b` is arguably the single most-called subroutine in
applied mathematics, and the story of how to do it *well* -- not just
correctly, but stably, in floating-point arithmetic, without needless
loss of precision -- runs from Gaussian elimination's two-century-old
roots through the mid-20th-century discovery that some numerically
"obvious" reformulations are secretly disastrous. This chronology traces
the decompositions and iterative solvers behind :mod:`mathematicskit.linalg`.

.. contents:: Timeline
   :local:
   :depth: 1

c. 150 CE -- The Nine Chapters and Elimination
-------------------------------------------------

The Chinese mathematical classic *Jiuzhang Suanshu* ("The Nine Chapters
on the Mathematical Art"), compiled from older material by the Han
dynasty, describes a method for solving simultaneous linear equations
by arranging their coefficients in a rectangular array and eliminating
unknowns column by column -- procedurally identical to what Gauss would
rediscover, and get his name attached to, sixteen centuries later.

*Implementation:* :func:`mathematicskit.numerical_analysis.systems.regression.PolynomialRegression`'s
underlying solves, and every dense linear system in this package, trace
back to exactly this elimination idea, now delegated to LAPACK via
:func:`scipy.linalg.lu`.

1809 -- Gauss and LU Decomposition
-------------------------------------

Carl Friedrich Gauss's elimination procedure for least-squares normal
equations (see the numerical-analysis chronology) factors as a byproduct
into a lower-triangular and an upper-triangular matrix, :math:`A = LU` --
a factorization made numerically robust by choosing, at each step, the
largest available pivot (Partial pivoting, formalized much later, in the
1940s-50s error-analysis literature). Once factored, solving for a new
right-hand side costs only :math:`O(n^2)`, not the :math:`O(n^3)` of a
fresh elimination.

*Implementation:* :func:`mathematicskit.linalg.systems.lu.lu_decompose` wraps
:func:`scipy.linalg.lu` for exactly this partial-pivoted factorization;
:func:`~mathematicskit.linalg.systems.lu.lu_solve_system` reuses it via
:func:`scipy.linalg.lu_factor`/:func:`~scipy.linalg.lu_solve`.

*References:* C. F. Gauss, *Theoria Motus Corporum Coelestium* (Hamburg,
1809).

.. minigallery:: ../../examples/linalg/lu/plot_01_lu_decomposition.py

1902 -- 1924 -- Cholesky's Decomposition
------------------------------------------

Andre-Louis Cholesky, an artillery officer and geodesist working on
triangulation surveys for the French army's Service Geographique,
devised around 1902-1910 a factorization specialized for exactly the
symmetric positive-definite systems that survey adjustment produces:
:math:`A = LL^T` with :math:`L` lower triangular, at roughly half the
arithmetic cost of a general LU factorization and without LU's need to
search for a pivot at all, since a symmetric positive-definite matrix's
own diagonal entries are always usable directly. Cholesky was killed in
action in 1918; his method was published posthumously in 1924 by
Commandant Benoit, from Cholesky's own manuscripts.

.. math::

   A = LL^T, \qquad A \text{ symmetric positive-definite}

*Implementation:* :func:`mathematicskit.linalg.systems.cholesky.cholesky_decompose`
wraps :func:`numpy.linalg.cholesky` for exactly this factorization, whose
failure (LAPACK's ``?potrf`` raising when a diagonal pivot would require
a square root of a negative number) doubles as a positive-definiteness
certificate in :func:`~mathematicskit.linalg.systems.cholesky.is_symmetric_positive_definite`;
:func:`~mathematicskit.linalg.systems.cholesky.cholesky_solve` reuses the
factorization for a forward/backward triangular solve via
:func:`scipy.linalg.solve_triangular`.

*References:* A.-L. Cholesky (posth.), "Sur la resolution numerique des
systemes d'equations lineaires," manuscript (c. 1910), transcribed and
published by Commandant Benoit, Bulletin Geodesique 2 (1924), 67-77.

.. minigallery:: ../../examples/linalg/cholesky/plot_01_cholesky_solve.py

1907 -- 1958 -- Gram-Schmidt, Householder, and QR
-----------------------------------------------------

Jorgen Pedersen Gram (1883) and Erhard Schmidt (1907) formalized the
process of building an orthonormal basis one vector at a time by
subtracting off each new vector's projection onto the vectors already
built -- a construction that, applied to a matrix's columns, factors it
as :math:`A = QR` with :math:`Q` orthonormal and :math:`R` upper
triangular. Alston Householder's 1958 paper gave a very different, far
more numerically stable route to the same factorization: a sequence of
elementary reflections that zero out sub-diagonal entries directly,
without ever explicitly re-orthogonalizing anything -- the textbook
demonstration that two mathematically equivalent formulas can behave
completely differently once floating-point rounding enters the picture.

*Implementation:* :func:`mathematicskit.linalg.systems.qr.householder_qr` wraps
:func:`scipy.linalg.qr`; :func:`~mathematicskit.linalg.systems.qr.gram_schmidt_qr`
keeps the classical/modified Gram-Schmidt construction hand-rolled
specifically to demonstrate, via
:func:`~mathematicskit.linalg.systems.qr.orthogonality_error`, how much
orthogonality classical Gram-Schmidt loses for an ill-conditioned matrix
that Householder's reflections do not.

*References:* E. Schmidt, "Zur Theorie der linearen und nichtlinearen
Integralgleichungen," Mathematische Annalen 63 (1907), 433-476; A. S.
Householder, "Unitary Triangularization of a Nonsymmetric Matrix,"
Journal of the ACM 5(4) (1958), 339-342.

.. minigallery:: ../../examples/linalg/qr/plot_01_householder_vs_gram_schmidt.py

1829 -- Cauchy and the Eigenvalue Problem
---------------------------------------------

Augustin-Louis Cauchy's 1829 memoir on the principal-axis problem for
quadratic forms proved that a real symmetric matrix has only real
eigenvalues and an orthogonal basis of eigenvectors -- the spectral
theorem, in essence, decades before "eigenvalue" (from the German
*Eigenwert*) entered the language via Hilbert at the turn of the 20th
century. Practical, efficient algorithms for computing those eigenvalues
-- rather than merely proving they exist -- would take more than a
century longer to arrive.

*Implementation:* :func:`mathematicskit.linalg.systems.eigen.eigen_symmetric`
wraps :func:`numpy.linalg.eigh` for exactly the symmetric case Cauchy's
theorem covers; :func:`~mathematicskit.linalg.systems.eigen.eigen_general` wraps
:func:`numpy.linalg.eig` for the general (possibly complex-spectrum)
case.

*References:* A.-L. Cauchy, "Sur l'équation à l'aide de laquelle on
détermine les inégalités séculaires des mouvements des planètes,"
Exercices de mathématiques 4 (1829).

1929 -- Power Iteration and von Mises
-----------------------------------------

Richard von Mises and Hilda Pollaczek-Geiringer's 1929 paper gave power
iteration its first systematic numerical treatment: repeatedly
multiplying a matrix into a vector and renormalizing converges to the
eigenvector of largest-magnitude eigenvalue, with the Rayleigh quotient
converging to the eigenvalue itself. Shifting the matrix before
inverting (inverse iteration) instead converges to whichever eigenvalue
sits nearest the shift -- both still the conceptually simplest route to
a single eigenpair when the full spectrum isn't needed, and the direct
ancestor of the shifted-QR algorithms that now compute the full spectrum
in production libraries.

*Implementation:* :func:`mathematicskit.linalg.systems.eigen.power_iteration`
and :func:`~mathematicskit.linalg.systems.eigen.inverse_iteration` implement
both exactly as originally described, the one family of algorithm in
this domain kept hand-rolled since the *iteration itself* is the point.

*References:* R. von Mises and H. Pollaczek-Geiringer, "Praktische
Verfahren der Gleichungsauflösung," Zeitschrift für Angewandte
Mathematik und Mechanik 9 (1929), 152-164, 58-77.

.. minigallery:: ../../examples/linalg/eigen/plot_01_eigenvalue_methods_compared.py

1936 -- Eckart, Young, and the Singular Value Decomposition
-----------------------------------------------------------------

Carl Eckart and Gale Young's 1936 paper (building on earlier work by
Beltrami, Jordan, and Sylvester on special cases) established the
general singular value decomposition :math:`A = U\Sigma V^T` for any
rectangular matrix, and proved the low-rank approximation theorem that
bears their names: truncating the SVD to its :math:`k` largest singular
values gives the *best possible* rank-:math:`k` approximation to
:math:`A` in either the spectral or Frobenius norm -- the theoretical
foundation under everything from principal component analysis to modern
recommender systems.

*Implementation:* :func:`mathematicskit.linalg.systems.svd.svd_decompose` wraps
:func:`numpy.linalg.svd`'s numerically stable bidiagonalization-based
algorithm, the modern production route rather than Eckart and Young's
own eigendecomposition-of-:math:`A^TA` construction (which squares the
condition number being worked with).

*References:* C. Eckart and G. Young, "The Approximation of One Matrix
by Another of Lower Rank," Psychometrika 1(3) (1936), 211-218.

.. minigallery:: ../../examples/linalg/svd/plot_01_svd.py

1947 -- 1948 -- Turing, von Neumann, and the Condition Number
--------------------------------------------------------------------

Alan Turing's 1948 paper "Rounding-Off Errors in Matrix Processes" and
John von Neumann and Herman Goldstine's 1947 study of Gaussian
elimination's numerical behavior founded rigorous backward-error
analysis for matrix computation, introducing what Turing called a
matrix's "N-condition number": a single number, built from the ratio of
its largest to smallest singular value, quantifying how much a linear
system's solution can be corrupted by small perturbations in its data --
independent of which algorithm later solves it. The same analysis
explained, for the first time, why some numerically "obvious"
reformulations (forming :math:`A^TA` explicitly for least squares,
rather than working with :math:`A` directly through its QR
factorization above) quietly discard accuracy that no amount of extra
floating-point precision recovers.

.. math::

   \kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}, \qquad
   \kappa_2(A^TA) = \kappa_2(A)^2

*Implementation:* :func:`mathematicskit.linalg.systems.stability.condition_number_2norm`
wraps :func:`numpy.linalg.cond` for exactly this ratio;
:func:`~mathematicskit.linalg.systems.stability.least_squares_normal_equations`
and :func:`~mathematicskit.linalg.systems.stability.least_squares_qr` solve the
same least-squares problem two different ways specifically to make the
squared-condition-number penalty of the normal equations concrete and
measurable side by side.

*References:* A. M. Turing, "Rounding-Off Errors in Matrix Processes,"
Quarterly Journal of Mechanics and Applied Mathematics 1(1) (1948),
287-308; J. von Neumann and H. H. Goldstine, "Numerical Inverting of
Matrices of High Order," Bulletin of the American Mathematical Society
53(11) (1947), 1021-1099.

.. minigallery:: ../../examples/linalg/stability/plot_01_normal_equations_vs_qr.py

1952 -- Hestenes, Stiefel, and the Conjugate Gradient Method
------------------------------------------------------------------

Magnus Hestenes and Eduard Stiefel's 1952 paper introduced the conjugate
gradient method for symmetric positive-definite systems: rather than
Gaussian elimination's direct :math:`O(n^3)` factorization, generate a
sequence of search directions mutually conjugate with respect to
:math:`A`, converging *exactly* within :math:`n` steps in exact
arithmetic and, in practice, far faster for well-conditioned or
clustered-spectrum matrices -- decades before "Krylov subspace methods"
became the standard name for the whole family it inaugurated. Yousef
Saad and Martin Schultz's 1986 GMRES extended the same Krylov-subspace
idea to general (non-symmetric) systems.

*Implementation:* :class:`mathematicskit.linalg.systems.iterative.ConjugateGradient`
and :class:`~mathematicskit.linalg.systems.iterative.GMRES` wrap
:func:`scipy.sparse.linalg.cg`/:func:`~scipy.sparse.linalg.gmres`
directly, adding the residual-history tracking (via each solver's
callback) that a bare library call doesn't expose.

*References:* M. R. Hestenes and E. Stiefel, "Methods of Conjugate
Gradients for Solving Linear Systems," Journal of Research of the
National Bureau of Standards 49(6) (1952), 409-436; Y. Saad and M. H.
Schultz, "GMRES: A Generalized Minimal Residual Algorithm for Solving
Nonsymmetric Linear Systems," SIAM Journal on Scientific and Statistical
Computing 7(3) (1986), 856-869.

.. minigallery:: ../../examples/linalg/iterative/plot_01_cg_and_gmres_convergence.py

See Also
--------

- :doc:`/api/linalg`
- :doc:`/history/numerical_analysis_breakthroughs`
- :doc:`/history/graph_theory_breakthroughs`
