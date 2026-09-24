Breakthroughs in Numerical Calculus
===================================


.. include:: /_generated/nav/calculus.rst

.. epigraph::

   "Nature laughs at the difficulties of integration."
   -- attributed to Pierre-Simon Laplace

The differential and integral calculus is by far the oldest subject in
this package, yet its *numerical* side is a comparatively young,
20th-century development: how to differentiate or integrate a function
you can only sample, or how to differentiate a computer program
directly. This chronology traces the ideas behind
:mod:`mathematicskit.calculus`, from the original calculus of Isaac
Newton and Gottfried Wilhelm Leibniz to the reverse-mode automatic
differentiation that now trains every large neural network.

.. contents:: Timeline
   :local:
   :depth: 1

1665-1684 -- Newton and Leibniz Invent the Calculus
---------------------------------------------------

Isaac Newton, working privately from about 1665, and Gottfried Wilhelm
Leibniz, working independently and publishing first in 1684, each
developed a complete differential and integral calculus. The result was
one of the bitterest priority disputes in the history of mathematics.
Leibniz's notation, :math:`dy/dx` and :math:`\int`, is the one still in
universal use; Newton's fluxion notation survives mainly in his
collected works. Both men agreed on the two central facts that every
numerical scheme in this module approximates: the derivative is a
limiting difference quotient, and the definite integral is a limiting
sum.

*Implementation:* :func:`mathematicskit.calculus.systems.finite_differences.central_difference`
approximates the limiting difference quotient at a finite step size;
:class:`mathematicskit.calculus.systems.quadrature.TrapezoidalRule`
approximates the limiting Riemann sum.

*References:* G. W. Leibniz, "Nova Methodus pro Maximis et Minimis,"
Acta Eruditorum (1684), 467-473.

.. minigallery:: ../../examples/calculus/finite_differences/plot_01_richardson_extrapolation.py

1715 -- Taylor's Theorem
------------------------

Brook Taylor's 1715 *Methodus Incrementorum Directa et Inversa* gave
the general expansion of a function as a power series built from its
derivatives at a single point. Colin Maclaurin's 1742 *Treatise of
Fluxions* used the special case expanded about zero so heavily that
this case now carries his name. Taylor's theorem underlies almost every
numerical approximation in this package: the error term of each
finite-difference formula, the convergence rate of each quadrature
rule, and the convergence order of each root-finder are all
Taylor-series arguments.

*Implementation:* :func:`mathematicskit.calculus.systems.taylor_series.maclaurin_coefficients`
and :func:`~mathematicskit.calculus.systems.taylor_series.taylor_remainder_bound`
implement the series and its Lagrange remainder bound.

*References:* B. Taylor, *Methodus Incrementorum Directa et Inversa*
(London, 1715); C. Maclaurin, *A Treatise of Fluxions* (Edinburgh,
1742).

.. minigallery:: ../../examples/calculus/taylor_series/plot_01_maclaurin_series.py

1814 -- Gauss and Gaussian Quadrature
-------------------------------------

Carl Friedrich Gauss's 1814 paper asked a sharp new question. If you
may choose *both* the sample points and their weights, rather than only
the weights as in Newton-Cotes rules, how accurate can an
:math:`n`-point quadrature rule be? Gauss's optimal nodes integrate
every polynomial up to degree :math:`2n-1` exactly, roughly twice the
degree an equally spaced rule with the same number of points achieves.
Carl Gustav Jacob Jacobi showed in 1826 that these nodes are the roots
of the degree-:math:`n` Legendre polynomial.

*Implementation:* :class:`mathematicskit.calculus.systems.quadrature.GaussianQuadrature`
wraps :func:`scipy.integrate.fixed_quad`, and
:func:`~mathematicskit.calculus.systems.quadrature.legendre_nodes_and_weights`
wraps :func:`numpy.polynomial.legendre.leggauss` for the underlying
node and weight computation.

*References:* C. F. Gauss, "Methodus Nova Integralium Valores per
Approximationem Inveniendi," Commentationes Societatis Regiae
Scientiarum Gottingensis Recentiores 3 (1814); C. G. J. Jacobi, "Ueber
Gauß' neue Methode, die Werthe der Integrale näherungsweise zu finden,"
Journal für die reine und angewandte Mathematik 1 (1826), 301-308.

.. minigallery:: ../../examples/calculus/quadrature/plot_01_rules_compared.py

1911-1927 -- Richardson Extrapolation
-------------------------------------

Lewis Fry Richardson is best known for his pioneering attempt at
numerical weather prediction, which was hopelessly slow to compute by
hand in his lifetime. In 1911, and more fully in a 1927 paper with
John Arthur Gaunt, he showed how to combine two estimates of the same
quantity, made at two different step sizes, so that their leading-order
error terms cancel. The combination is more accurate than either
estimate and costs no extra computation. Applied to a central-difference
derivative, or to the trapezoidal rule (where it is called Romberg
integration), Richardson extrapolation is the standard way to gain an
extra order or two of accuracy almost for free.

*Implementation:* :func:`mathematicskit.calculus.systems.finite_differences.richardson_extrapolation`
implements this repeated-halving, error-cancelling extrapolation as a
triangular (Neville-style) table.

*References:* L. F. Richardson, "The Approximate Arithmetical Solution
by Finite Differences of Physical Problems Involving Differential
Equations," Philosophical Transactions of the Royal Society A 210
(1911), 307-357; L. F. Richardson and J. A. Gaunt, "The Deferred
Approach to the Limit," Philosophical Transactions of the Royal Society
A 226 (1927), 299-361.

.. minigallery:: ../../examples/calculus/finite_differences/plot_01_richardson_extrapolation.py

1964-1970 -- Wengert, Linnainmaa, and Automatic Differentiation
---------------------------------------------------------------

Robert Wengert's 1964 paper described forward-mode automatic
differentiation: every arithmetic operation also carries its derivative,
propagated alongside the value by the chain rule. This is exactly what
happens when ordinary arithmetic is performed on a "dual number"
:math:`a + b\varepsilon` with :math:`\varepsilon^2=0`. Seppo
Linnainmaa's 1970 master's thesis described the complementary reverse
mode, which propagates derivatives *backward* through a computational
graph after a forward pass. Reverse mode computes the gradient with
respect to any number of inputs at a small constant multiple of the
cost of one forward evaluation, whatever the input dimension.
Rediscovered and popularized as "backpropagation" in the
neural-network literature of the 1980s, it is the algorithm that makes
training a modern large neural network feasible at all.

*Implementation:* :class:`mathematicskit.calculus.systems.dual_numbers.Dual`
implements forward-mode automatic differentiation with dual-number
arithmetic; :class:`~mathematicskit.calculus.systems.autodiff.Variable`
and :func:`~mathematicskit.calculus.systems.autodiff.gradient` implement
reverse-mode automatic differentiation over a small computational graph.

*References:* R. E. Wengert, "A Simple Automatic Derivative Evaluation
Program," Communications of the ACM 7(8) (1964), 463-464; S. Linnainmaa,
"The Representation of the Cumulative Rounding Error of an Algorithm as
a Taylor Expansion of the Local Rounding Errors" (Master's thesis,
University of Helsinki, 1970; in Finnish).

.. minigallery:: ../../examples/calculus/dual_numbers/plot_01_forward_mode_autodiff.py

.. minigallery:: ../../examples/calculus/autodiff/plot_01_reverse_mode_gradients.py

1983 -- QUADPACK and Adaptive Quadrature
----------------------------------------

Adaptive quadrature does not fix the number of subdivisions in advance.
Instead it subdivides *where the integrand needs it*: estimate an
interval's integral two ways, and if the estimates disagree by more
than a tolerance, split the interval and recurse. Robert Piessens, Elise
de Doncker-Kapenga, Christoph Überhuber, and David Kahaner developed the
QUADPACK library through the 1970s and published it as a book in 1983.
It gave the idea its definitive, extensively validated implementation,
which SciPy still ships essentially unchanged.

*Implementation:* :class:`mathematicskit.calculus.systems.quadrature.AdaptiveQuadrature`
wraps :func:`scipy.integrate.quad`, which runs QUADPACK's ``QAGS``
routine under the hood.

*References:* R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber, and
D. K. Kahaner, *QUADPACK: A Subroutine Package for Automatic
Integration* (Berlin: Springer, 1983).

.. minigallery:: ../../examples/calculus/quadrature/plot_01_rules_compared.py

See Also
--------

- :doc:`/api/calculus`
- :doc:`/history/numerical_analysis_breakthroughs`
- :doc:`/history/special_functions_breakthroughs`
