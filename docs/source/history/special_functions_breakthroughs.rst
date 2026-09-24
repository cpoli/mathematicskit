Breakthroughs in Special Functions and Transforms
=================================================


.. include:: /_generated/nav/special_functions.rst

.. epigraph::

   "I have had my results for a long time, but I do not yet know how I
   am to arrive at them."
   -- attributed to Carl Friedrich Gauss

The "special functions" -- gamma, Bessel, the orthogonal polynomial
families -- are special because they keep recurring, unbidden, as exact
solutions of a huge range of unrelated differential equations, from
vibrating drumheads to quantum wavefunctions. This chronology traces the
ideas behind :mod:`mathematicskit.special_functions`, from Leonhard
Euler's extension of the factorial to the fast Fourier transform, an
algorithm rediscovered from a much older idea that made digital signal
processing computationally feasible.

.. contents:: Timeline
   :local:
   :depth: 1

1729 -- Euler's Gamma Function
------------------------------

Christian Goldbach asked how the factorial could be extended to
non-integer values. Leonhard Euler answered in an October 1729 letter
with an infinite-product formula for such an interpolation, and gave the
integral form early the next year. Adrien-Marie Legendre later named the
function :math:`\Gamma`. It satisfies :math:`\Gamma(n) = (n-1)!` at
positive integers and extends smoothly everywhere else, with poles at
zero and the negative integers. The closely related beta integral
:math:`B(a,b)`, which Euler also studied and Jacques Binet named in
1839, can be written directly in terms of :math:`\Gamma`.

.. math::

   \Gamma(x) = \int_0^\infty t^{x-1}e^{-t}\,dt, \qquad
   B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}

*Implementation:* :func:`mathematicskit.special_functions.systems.gamma_beta.gamma_function`
and :func:`~mathematicskit.special_functions.systems.gamma_beta.beta_function`
wrap :func:`scipy.special.gamma`/:func:`~scipy.special.beta` directly.
:func:`~mathematicskit.special_functions.systems.gamma_beta.log_gamma_function`
wraps the numerically stable :func:`scipy.special.gammaln` for arguments
large enough that :math:`\Gamma` itself would overflow.

*References:* L. Euler, letter to C. Goldbach, 13 October 1729, in P.-H.
Fuss, ed., *Correspondance mathématique et physique de quelques célèbres
géomètres du XVIIIème siècle*, vol. 1 (St. Petersburg, 1843), 56-60.

.. minigallery:: ../../examples/special_functions/gamma_beta/plot_01_gamma_extends_factorial.py

1782-1854 -- Legendre and Chebyshev Polynomials
-----------------------------------------------

Adrien-Marie Legendre's 1782 work on the gravitational attraction of
spheroids introduced the polynomial family that now bears his name. The
polynomials solve Laplace's equation in spherical coordinates and are
orthogonal on :math:`[-1,1]` with the simplest possible weight, 1.
Pafnuty Chebyshev's 1854 memoir on least-deviation approximation (see
the numerical-analysis chronology) introduced a second family,
orthogonal with respect to the weight :math:`1/\sqrt{1-x^2}` and given
in closed form by :math:`T_n(x) = \cos(n\arccos x)`. Its
minimal-oscillation property is exactly what makes Chebyshev *nodes*
the cure for Runge's phenomenon.

*Implementation:* :func:`mathematicskit.special_functions.systems.orthogonal_polynomials.legendre_polynomial`
and :func:`~mathematicskit.special_functions.systems.orthogonal_polynomials.chebyshev_polynomial`
wrap :func:`numpy.polynomial.legendre.legval`/:func:`~numpy.polynomial.chebyshev.chebval`
directly, and their orthogonality is verified numerically with
:func:`mathematicskit.special_functions.utils.orthogonality.inner_product`.

*References:* A.-M. Legendre, "Recherches sur l'attraction des
sphéroïdes homogènes," Mémoires de Mathématique et de Physique,
présentés à l'Académie Royale des Sciences 10 (1785), 411-434
(presented 1782).

.. minigallery:: ../../examples/special_functions/orthogonal_polynomials/plot_01_four_families.py

1810-1879 -- Hermite and Laguerre Polynomials
---------------------------------------------

Pierre-Simon Laplace in 1810, and Pafnuty Chebyshev more systematically
in 1859, studied polynomials orthogonal with respect to the Gaussian
weight :math:`e^{-x^2}`. Charles Hermite's 1864 memoir gave them their
definitive treatment and the name they carry today. Edmond Laguerre's
1879 paper introduced the complementary family, orthogonal on the
half-line :math:`[0,\infty)` with weight :math:`e^{-x}`. Nearly half a
century later, both families turned out to be exactly the eigenfunctions
quantum mechanics needed: Hermite polynomials for the quantum harmonic
oscillator, and Laguerre polynomials for the radial part of the
hydrogen atom's wavefunctions.

*Implementation:* :func:`mathematicskit.special_functions.systems.orthogonal_polynomials.hermite_polynomial`
and :func:`~mathematicskit.special_functions.systems.orthogonal_polynomials.laguerre_polynomial`
wrap :func:`numpy.polynomial.hermite.hermval`/:func:`~numpy.polynomial.laguerre.lagval`.
Their orthogonality against each family's weight is verified
numerically with :func:`scipy.integrate.quad` over an infinite domain.

*References:* C. Hermite, "Sur un nouveau développement en série des
fonctions," Comptes Rendus de l'Académie des Sciences 58 (1864), 93-100,
266-273; E. Laguerre, "Sur l'intégrale
:math:`\int_x^\infty e^{-x}x^{-1}dx`," Bulletin de la Société
Mathématique de France 7 (1879), 72-81.

.. minigallery:: ../../examples/special_functions/orthogonal_polynomials/plot_01_four_families.py

1822-1965 -- Fourier's Theorem and the Fast Fourier Transform
-------------------------------------------------------------

Joseph Fourier's 1822 treatise on heat conduction claimed,
controversially at the time, that essentially any periodic function can
be written as a sum of sines and cosines. The claim, first presented to
the Paris Academy in 1807, was made rigorous only gradually over the
following century, but it became the conceptual foundation of the
discrete Fourier transform once digital computers made the transform
something to actually *compute*. James Cooley and John Tukey's 1965
paper showed how to compute a length-:math:`N` discrete Fourier
transform recursively from two length-:math:`N/2` transforms, cutting
the cost from :math:`O(N^2)` to :math:`O(N\log N)`. They had largely
independently rediscovered an algorithm that Carl Friedrich Gauss
sketched in an unpublished 1805 manuscript on interpolating asteroid
orbits. For the signal lengths that radar and telecommunications
needed, the speedup was so large that the algorithm is routinely
credited with making digital signal processing practical.

.. math::

   X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi ikn/N}

*Implementation:* :func:`mathematicskit.special_functions.systems.fourier_transform.fft_numpy`
wraps :func:`numpy.fft.fft` as the primary API.
:func:`~mathematicskit.special_functions.systems.fourier_transform.dft_naive`
and :func:`~mathematicskit.special_functions.systems.fourier_transform.fft_radix2`
hand-roll the naive :math:`O(N^2)` sum and the Cooley-Tukey radix-2
recursion. :func:`~mathematicskit.special_functions.systems.fourier_transform.compare_fft_methods`
times all three against each other to make the :math:`O(N^2)` versus
:math:`O(N\log N)` gap visible. This is the one submodule in this
domain where hand-rolling, rather than a direct library call, is the
point.

*References:* J. Fourier, *Théorie analytique de la chaleur* (Paris:
Firmin Didot, 1822); J. W. Cooley and J. W. Tukey, "An Algorithm for the
Machine Calculation of Complex Fourier Series," Mathematics of
Computation 19(90) (1965), 297-301; M. T. Heideman, D. H. Johnson, and
C. S. Burrus, "Gauss and the History of the Fast Fourier Transform,"
Archive for History of Exact Sciences 34(3) (1985), 265-277
(documenting Gauss's unpublished 1805 anticipation).

.. minigallery:: ../../examples/special_functions/fourier_transform/plot_01_naive_vs_radix2_vs_numpy.py

1824 -- Bessel Functions
------------------------

In the 1730s, Daniel Bernoulli's study of the oscillations of a hanging
chain had already produced solutions of what is now recognized as a
special case of Bessel's equation. Friedrich Wilhelm Bessel's 1824
memoir on planetary perturbations gave the family its first systematic
treatment and tabulation. His motivation was the periodic problem of
Kepler's equation for planetary orbits. :math:`J_\nu`, regular at the
origin, and :math:`Y_\nu`, singular there and introduced later by Carl
Neumann and Heinrich Weber, are two linearly independent solutions of
the same second-order equation. They appear throughout physics and
engineering wherever a problem has cylindrical symmetry: vibrating
drumheads, waveguides, heat conduction in a cylinder.

.. math::

   x^2 y'' + xy' + (x^2 - \nu^2)y = 0

*Implementation:* :func:`mathematicskit.special_functions.systems.bessel.bessel_first_kind`
and :func:`~mathematicskit.special_functions.systems.bessel.bessel_second_kind`
wrap :func:`scipy.special.jv`/:func:`~scipy.special.yv` directly. This
domain's tests check them against the defining differential equation
itself, using finite differences.

*References:* F. W. Bessel, "Untersuchung des Theils der planetarischen
Störungen, welcher aus der Bewegung der Sonne entsteht," Abhandlungen
der Königlichen Akademie der Wissenschaften zu Berlin (1824), 1-52.

.. minigallery:: ../../examples/special_functions/bessel/plot_01_bessel_functions.py

See Also
--------

- :doc:`/api/special_functions`
- :doc:`/history/numerical_analysis_breakthroughs`
- :doc:`/history/calculus_breakthroughs`
