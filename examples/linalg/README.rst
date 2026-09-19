Examples
========

This gallery walks through every public feature of ``mathkit.linalg``:
LU/QR/Cholesky decompositions, symmetric eigenvalue algorithms, SVD,
iterative Krylov solvers, and least-squares stability.

See also the narrative tutorial:

- :doc:`/tutorials/eigenvalues_everywhere`

Each script in this gallery is self-contained and can be run directly with
``python examples/linalg/<section>/<script>.py``.

Sections
--------

- **lu** -- LU decomposition with partial pivoting, and solving/computing
  determinants from it.
- **qr** -- Householder vs. Gram-Schmidt QR, and why Householder stays
  orthogonal for ill-conditioned matrices.
- **cholesky** -- Cholesky decomposition for SPD matrices.
- **eigen** -- eigenvalue computation via numpy.linalg.eigh/eig, power
  iteration, and inverse iteration.
- **svd** -- singular value decomposition via numpy.linalg.svd.
- **iterative** -- conjugate gradient and GMRES convergence.
- **stability** -- least squares via normal equations vs. QR, and why the
  former squares the condition number.
