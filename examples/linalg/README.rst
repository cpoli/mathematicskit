Examples
========

This gallery walks through every public feature of ``mathematicskit.linalg``:
LU/QR/Cholesky decompositions, symmetric eigenvalue algorithms, SVD,
iterative Krylov solvers, and least-squares stability.

See also the narrative tutorial:

- :doc:`/tutorials/eigenvalues_everywhere`

Each script in this gallery is self-contained and can be run directly with
``python examples/linalg/<section>/<script>.py``.

Sections
--------

- **classical** -- Cramer's rule and the Cayley-Hamilton theorem.
- **lu** -- LU decomposition with partial pivoting, and solving/computing
  determinants from it.
- **qr** -- Householder vs. Gram-Schmidt QR, and why Householder stays
  orthogonal for ill-conditioned matrices.
- **cholesky** -- Cholesky decomposition for SPD matrices.
- **eigen** -- eigenvalue computation via numpy.linalg.eigh/eig, power
  iteration, inverse iteration, Gershgorin discs, Lanczos, and the QR
  algorithm / Schur form.
- **svd** -- singular value decomposition via numpy.linalg.svd, and the
  Moore-Penrose pseudoinverse.
- **iterative** -- conjugate gradient and GMRES convergence; Jacobi,
  Gauss-Seidel, and SOR.
- **stability** -- least squares via normal equations vs. QR, and why the
  former squares the condition number.
