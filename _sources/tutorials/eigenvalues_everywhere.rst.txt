:orphan:

Eigenvalues Everywhere
========================

The eigenvalue problem :math:`Av = \lambda v` is one of the few pieces
of mathematics that shows up, essentially unchanged, in three
completely unrelated corners of mathkit: as the core linear-algebra
decomposition in :mod:`mathkit.linalg`, as the tool that reveals a
graph's community structure in :mod:`mathkit.graph_theory`, and as the
classification scheme for a dynamical system's fixed points in
:mod:`mathkit.ode_dynamics`.

Eigenvalues as a decomposition
------------------------------------

:func:`~mathkit.linalg.systems.eigen.eigen_symmetric` computes the full
eigendecomposition of a symmetric matrix directly:

.. code-block:: python

   import numpy as np
   from mathkit.linalg import eigen_symmetric

   A = np.array([[2.0, 1.0], [1.0, 2.0]])
   result = eigen_symmetric(A)
   print(result.eigenvalues)
   # [1. 3.]

Eigenvalues as graph structure
------------------------------------

A graph's Laplacian :math:`L = D - A` (degree matrix minus adjacency
matrix) is always symmetric positive semi-definite, so
:func:`~mathkit.graph_theory.systems.spectral.spectral_analysis` can
apply the very same symmetric eigendecomposition to it. For a 6-cycle,
the Laplacian eigenvalues have the closed form :math:`2 - 2\cos(2\pi
k/6)`:

.. code-block:: python

   from mathkit.graph_theory import spectral_analysis
   from mathkit.graph_theory.utils.generators import cycle_graph

   g = cycle_graph(6)
   spec = spectral_analysis(g)
   print(np.round(spec.eigenvalues, 6))
   # [0. 1. 1. 3. 3. 4.]
   print(spec.algebraic_connectivity)
   # 0.9999999999999994 -- the second-smallest eigenvalue, positive since the graph is connected

The smallest eigenvalue is always exactly 0 (the all-ones vector is
always an eigenvector); the second-smallest -- the algebraic
connectivity -- is positive if and only if the graph is connected at
all, and the corresponding eigenvector's sign pattern is exactly what
:func:`~mathkit.graph_theory.visualizers.plots.plot_spectral_bipartition`
uses to split the graph into two communities.

Eigenvalues as stability classification
---------------------------------------------

A 2D linear system's fixed point is classified entirely by its
Jacobian's eigenvalues:
:func:`~mathkit.ode_dynamics.systems.stability.classify_fixed_point_2d`
computes them via the closed-form trace-determinant formula (equivalent
to :func:`numpy.linalg.eigvals`, just without forming the general
complex-eigenvalue machinery for a case this simple) and reads off the
qualitative behavior directly from their sign:

.. code-block:: python

   from mathkit.ode_dynamics.systems.stability import classify_fixed_point_2d

   J = np.array([[-1.0, 0.0], [0.0, -2.0]])
   result = classify_fixed_point_2d(J)
   print(result.classification, result.eigenvalues)
   # stable node [-1.+0.j -2.+0.j]

Both eigenvalues negative means every nearby trajectory decays toward
the fixed point -- a stable node, the same qualitative conclusion
:mod:`mathkit.linalg`'s eigenvalue sign would give for the stability of
any linear system :math:`\dot x = Jx`, and the same computation
:mod:`mathkit.graph_theory` uses to certify a graph is connected, just
applied to a different matrix each time.

See Also
--------

- :doc:`/api/linalg`
- :doc:`/api/graph_theory`
- :doc:`/api/ode_dynamics`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/graph_theory_breakthroughs`