Breakthroughs in Numerical Linear Algebra
=========================================


.. include:: /_generated/nav/linalg.rst

.. epigraph::

   "Linear algebra is the ultimate service subject."
   -- attributed to Gilbert Strang

Solving :math:`Ax=b` is arguably the most frequently called subroutine
in applied mathematics. The story of how to do it *well* -- not just
correctly, but stably in floating-point arithmetic, without needless
loss of precision -- runs from the ancient roots of elimination to the
mid-20th-century discovery that some "obvious" reformulations are
numerically disastrous. This chronology traces the decompositions and
iterative solvers behind :mod:`mathematicskit.linalg`.

.. contents:: Timeline
   :local:
   :depth: 1

c. 100 CE -- The Nine Chapters and Elimination
----------------------------------------------

The Chinese classic *Jiuzhang Suanshu* ("The Nine Chapters on the
Mathematical Art") was compiled from older material and reached its
final form by around the 1st century CE. Its eighth chapter solves
simultaneous linear equations by arranging their coefficients in a
rectangular array and eliminating unknowns column by column. The
procedure is essentially the one Isaac Newton described in Europe in
1707, and it became attached to Carl Friedrich Gauss's name after his
work on least squares, some seventeen centuries after the *Nine
Chapters*.

*Implementation:* :func:`mathematicskit.numerical_analysis.systems.regression.PolynomialRegression`'s
underlying solves, and every dense linear system in this package, rest
on this elimination idea, now delegated to LAPACK through
:func:`scipy.linalg.lu`.

.. minigallery:: ../../examples/linalg/lu/plot_01_lu_decomposition.py

1809 -- Gauss and LU Decomposition
----------------------------------

Carl Friedrich Gauss's elimination procedure for the least-squares
normal equations (see the numerical-analysis chronology) implicitly
factors the matrix into a lower-triangular and an upper-triangular
matrix, :math:`A = LU`. The factorization is made numerically robust by
choosing the largest available pivot at each step; this partial
pivoting was analyzed only much later, in the error-analysis literature
of the 1940s and 1950s. Once :math:`A` is factored, solving for a new
right-hand side costs only :math:`O(n^2)` operations instead of the
:math:`O(n^3)` of a fresh elimination.

*Implementation:* :func:`mathematicskit.linalg.systems.lu.lu_decompose`
wraps :func:`scipy.linalg.lu` for this partial-pivoted factorization;
:func:`~mathematicskit.linalg.systems.lu.lu_solve_system` reuses it via
:func:`scipy.linalg.lu_factor`/:func:`~scipy.linalg.lu_solve`.

*References:* C. F. Gauss, *Theoria Motus Corporum Coelestium* (Hamburg:
Perthes et Besser, 1809).

.. minigallery:: ../../examples/linalg/lu/plot_01_lu_decomposition.py

1829 -- Cauchy and the Eigenvalue Problem
-----------------------------------------

Augustin-Louis Cauchy's 1829 memoir on the principal axes of quadratic
forms proved that a real symmetric matrix has only real eigenvalues and
an orthogonal basis of eigenvectors. This is the spectral theorem in
essence, written decades before the word "eigenvalue" (from David
Hilbert's German *Eigenwert*) entered the language around 1904. Carl
Gustav Jacob Jacobi's 1846 rotation method aside, practical algorithms
for *computing* eigenvalues, rather than proving they exist, had to
wait for the computer era.

*Implementation:* :func:`mathematicskit.linalg.systems.eigen.eigen_symmetric`
wraps :func:`numpy.linalg.eigh` for the symmetric case that Cauchy's
theorem covers; :func:`~mathematicskit.linalg.systems.eigen.eigen_general`
wraps :func:`numpy.linalg.eig` for the general case, whose spectrum may
be complex.

*References:* A.-L. Cauchy, "Sur l'équation à l'aide de laquelle on
détermine les inégalités séculaires des mouvements des planètes,"
Exercices de mathématiques 4 (1829).

.. minigallery:: ../../examples/linalg/eigen/plot_01_eigenvalue_methods_compared.py

1883-1958 -- Gram-Schmidt, Householder, and QR
----------------------------------------------

Jørgen Pedersen Gram (1883) and Erhard Schmidt (1907) formalized the
process of building an orthonormal basis one vector at a time, by
subtracting from each new vector its projections onto the vectors
already built. Applied to a matrix's columns, the process factors the
matrix as :math:`A = QR`, with :math:`Q` orthonormal and :math:`R`
upper triangular. Alston Householder's 1958 paper gave a very different
and far more numerically stable route to the same factorization: a
sequence of reflections that zero out the entries below the diagonal
directly, with no re-orthogonalization at all. The pair is the textbook
demonstration that two mathematically equivalent procedures can behave
completely differently once floating-point rounding enters.

*Implementation:* :func:`mathematicskit.linalg.systems.qr.householder_qr`
wraps :func:`scipy.linalg.qr`.
:func:`~mathematicskit.linalg.systems.qr.gram_schmidt_qr` keeps the
classical and modified Gram-Schmidt constructions hand-rolled to show,
via :func:`~mathematicskit.linalg.systems.qr.orthogonality_error`, how
much orthogonality classical Gram-Schmidt loses on an ill-conditioned
matrix compared with Householder's reflections.

*References:* J. P. Gram, "Ueber die Entwickelung reeller Functionen in
Reihen mittelst der Methode der kleinsten Quadrate," Journal für die
reine und angewandte Mathematik 94 (1883), 41-73; E. Schmidt, "Zur
Theorie der linearen und nichtlinearen Integralgleichungen,"
Mathematische Annalen 63 (1907), 433-476; A. S. Householder, "Unitary
Triangularization of a Nonsymmetric Matrix," Journal of the ACM 5(4)
(1958), 339-342.

.. minigallery:: ../../examples/linalg/qr/plot_01_householder_vs_gram_schmidt.py

1910-1924 -- Cholesky's Decomposition
-------------------------------------

André-Louis Cholesky was an artillery officer and geodesist who worked
on triangulation surveys for the French army's Service géographique.
By 1910 he had devised a factorization for the symmetric
positive-definite systems that survey adjustment produces:
:math:`A = LL^T`, with :math:`L` lower triangular. It costs roughly
half as much as a general LU factorization and needs no pivot search,
because the diagonal entries of a symmetric positive-definite matrix
can always be used directly. Cholesky was killed in action in 1918;
Commandant Benoit published the method from his manuscripts in 1924.

.. math::

   A = LL^T, \qquad A \text{ symmetric positive-definite}

*Implementation:* :func:`mathematicskit.linalg.systems.cholesky.cholesky_decompose`
wraps :func:`numpy.linalg.cholesky` for this factorization. Its failure
(LAPACK's ``?potrf`` raises an error when a pivot would require the
square root of a negative number) doubles as a positive-definiteness
test in
:func:`~mathematicskit.linalg.systems.cholesky.is_symmetric_positive_definite`.
:func:`~mathematicskit.linalg.systems.cholesky.cholesky_solve` reuses the
factorization for a forward and backward triangular solve with
:func:`scipy.linalg.solve_triangular`.

*References:* Commandant Benoit, "Note sur une méthode de résolution des
équations normales provenant de l'application de la méthode des
moindres carrés à un système d'équations linéaires en nombre inférieur
à celui des inconnues (Procédé du Commandant Cholesky)," Bulletin
Géodésique 2 (1924), 67-77; A.-L. Cholesky, "Sur la résolution
numérique des systèmes d'équations linéaires" (manuscript dated 1910).

.. minigallery:: ../../examples/linalg/cholesky/plot_01_cholesky_solve.py

1929 -- Power Iteration and von Mises
-------------------------------------

Richard von Mises and Hilda Pollaczek-Geiringer's 1929 paper gave power
iteration its first systematic numerical treatment. Repeatedly
multiplying a vector by a matrix and renormalizing converges to the
eigenvector of the largest-magnitude eigenvalue, and the Rayleigh
quotient converges to that eigenvalue. Inverse iteration, which Helmut
Wielandt developed in 1944, applies the same idea to the inverse of a
shifted matrix and converges to whichever eigenvalue lies nearest the
shift. Both remain the simplest route to a single eigenpair when the
full spectrum isn't needed, and both are ancestors of the shifted QR
algorithm that production libraries now use for the full spectrum.

*Implementation:* :func:`mathematicskit.linalg.systems.eigen.power_iteration`
and :func:`~mathematicskit.linalg.systems.eigen.inverse_iteration`
implement both as originally described. They are the one algorithm
family in this domain kept hand-rolled, because the *iteration itself*
is the point.

*References:* R. von Mises and H. Pollaczek-Geiringer, "Praktische
Verfahren der Gleichungsauflösung," Zeitschrift für Angewandte
Mathematik und Mechanik 9 (1929), 58-77, 152-164.

.. minigallery:: ../../examples/linalg/eigen/plot_01_eigenvalue_methods_compared.py

1936 -- Eckart, Young, and the Singular Value Decomposition
-----------------------------------------------------------

Carl Eckart and Gale Young's 1936 paper built on earlier special cases
by Eugenio Beltrami, Camille Jordan, and James Joseph Sylvester. It
applied the singular value decomposition :math:`A = U\Sigma V^T` to
general rectangular matrices and proved the low-rank approximation
theorem that bears their names: truncating the SVD to its :math:`k`
largest singular values gives the *best possible* rank-:math:`k`
approximation to :math:`A` in the Frobenius norm. Leon Mirsky extended
the result in 1960 to every unitarily invariant norm, including the
spectral norm. The theorem underlies everything from principal
component analysis to modern recommender systems.

*Implementation:* :func:`mathematicskit.linalg.systems.svd.svd_decompose`
wraps the numerically stable, bidiagonalization-based algorithm in
:func:`numpy.linalg.svd`. This is the modern production route, not
Eckart and Young's construction through the eigendecomposition of
:math:`A^TA`, which squares the condition number.

*References:* C. Eckart and G. Young, "The Approximation of One Matrix
by Another of Lower Rank," Psychometrika 1(3) (1936), 211-218; L.
Mirsky, "Symmetric Gauge Functions and Unitarily Invariant Norms,"
Quarterly Journal of Mathematics 11(1) (1960), 50-59.

.. minigallery:: ../../examples/linalg/svd/plot_01_svd.py

1947-1948 -- Turing, von Neumann, and the Condition Number
----------------------------------------------------------

John von Neumann and Herman Goldstine's 1947 study of Gaussian
elimination and Alan Turing's 1948 paper "Rounding-Off Errors in Matrix
Processes" founded the rigorous rounding-error analysis of matrix
computations. Turing coined the term "condition number" for a single
number that measures how much a linear system's solution can be
corrupted by small perturbations in its data, whichever algorithm
solves it. Turing's own versions were built from matrix norms; the
standard modern 2-norm version is the ratio of the largest to the
smallest singular value. The same analysis explained why some
"obvious" reformulations lose accuracy that no amount of extra
floating-point precision recovers. A key example is forming
:math:`A^TA` explicitly for least squares, rather than working with
:math:`A` directly through its QR factorization.

.. math::

   \kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}, \qquad
   \kappa_2(A^TA) = \kappa_2(A)^2

*Implementation:* :func:`mathematicskit.linalg.systems.stability.condition_number_2norm`
wraps :func:`numpy.linalg.cond` for this ratio.
:func:`~mathematicskit.linalg.systems.stability.least_squares_normal_equations`
and :func:`~mathematicskit.linalg.systems.stability.least_squares_qr`
solve the same least-squares problem two ways, so that the
squared-condition-number penalty of the normal equations can be
measured side by side.

*References:* J. von Neumann and H. H. Goldstine, "Numerical Inverting
of Matrices of High Order," Bulletin of the American Mathematical
Society 53(11) (1947), 1021-1099; A. M. Turing, "Rounding-Off Errors in
Matrix Processes," Quarterly Journal of Mechanics and Applied
Mathematics 1(1) (1948), 287-308.

.. minigallery:: ../../examples/linalg/stability/plot_01_normal_equations_vs_qr.py

1952 -- Hestenes, Stiefel, and the Conjugate Gradient Method
------------------------------------------------------------

Magnus Hestenes and Eduard Stiefel's 1952 paper introduced the conjugate
gradient method for symmetric positive-definite systems. Instead of
Gaussian elimination's direct :math:`O(n^3)` factorization, it
generates a sequence of search directions that are mutually conjugate
with respect to :math:`A`. In exact arithmetic it converges within
:math:`n` steps, and in practice far sooner for well-conditioned
matrices or matrices with clustered eigenvalues. It launched the family
now called Krylov subspace methods, decades before that name became
standard. Yousef Saad and Martin Schultz's 1986 GMRES extended the same
Krylov-subspace idea to general non-symmetric systems.

*Implementation:* :class:`mathematicskit.linalg.systems.iterative.ConjugateGradient`
and :class:`~mathematicskit.linalg.systems.iterative.GMRES` wrap
:func:`scipy.sparse.linalg.cg`/:func:`~scipy.sparse.linalg.gmres`
directly. They add residual-history tracking through each solver's
callback, which a bare library call doesn't expose.

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
