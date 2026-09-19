Breakthroughs in Special Functions and Transforms
=====================================================


.. include:: /_generated/nav/special_functions.rst

.. epigraph::

   "I have had my results for a long time, but I do not yet know how I
   am to arrive at them." -- Carl Friedrich Gauss, on withholding
   publication of results he could not yet prove rigorously, as recorded
   by Wolfgang Sartorius von Waltershausen

The "special functions" -- gamma, Bessel, the orthogonal polynomial
families -- are special precisely because they recur, unbidden, as the
exact solutions to an enormous range of unrelated differential
equations, from vibrating drumheads to quantum wavefunctions. This
chronology traces the ideas behind :mod:`mathkit.special_functions`, from
Euler's extension of the factorial to the algorithm, rediscovered from a
much older idea, that made digital signal processing computationally
feasible at all.

.. contents:: Timeline
   :local:
   :depth: 1

1729 -- Euler's Gamma Function
-------------------------------------

Leonhard Euler, answering a question posed by Christian Goldbach about
interpolating the factorial to non-integer values, gave in an October
1729 letter an infinite-product formula for exactly such an
interpolation -- the function later named :math:`\Gamma` by Adrien-Marie
Legendre, satisfying :math:`\Gamma(n) = (n-1)!` at positive integers and
extending smoothly (and, at negative integers, with poles) everywhere
else. Legendre's 1811 companion function :math:`B(a,b)`, expressible
directly in terms of :math:`\Gamma`, arose from his own studies of
elliptic integrals.

.. math::

   \Gamma(x) = \int_0^\infty t^{x-1}e^{-t}\,dt, \qquad
   B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}

*Implementation:* :func:`mathkit.special_functions.systems.gamma_beta.gamma_function`
and :func:`~mathkit.special_functions.systems.gamma_beta.beta_function`
wrap :func:`scipy.special.gamma`/:func:`~scipy.special.beta` directly;
:func:`~mathkit.special_functions.systems.gamma_beta.log_gamma_function`
wraps the numerically stable :func:`scipy.special.gammaln` for
arguments large enough that :math:`\Gamma` itself would overflow.

*References:* L. Euler, letter to C. Goldbach, 13 October 1729, in P.-H.
Fuss, ed., *Correspondance Mathematique et Physique de Quelques Celebres
Geometres du XVIIIeme Siecle*, vol. 1 (St. Petersburg, 1843), 56-60.

.. minigallery:: ../../examples/special_functions/gamma_beta/plot_01_gamma_extends_factorial.py

1817 -- 1824 -- Bessel Functions
---------------------------------------

Daniel Bernoulli had already encountered, in his 1730s studies of a
hanging chain's oscillation, solutions of what is now recognized as a
special case of Bessel's equation; Friedrich Wilhelm Bessel's 1824
memoir on planetary perturbations gave the general family of solutions
its first systematic treatment and tabulation, motivated by exactly the
kind of periodic, cylindrically symmetric problem (here, Kepler's
equation) where these functions arise naturally. :math:`J_\nu`, regular
at the origin, and :math:`Y_\nu`, singular there, are the two linearly
independent solutions of the same second-order equation, and now appear
throughout physics and engineering wherever a problem has cylindrical
symmetry: vibrating drumheads, waveguides, heat conduction in a
cylinder.

.. math::

   x^2 y'' + xy' + (x^2 - \nu^2)y = 0

*Implementation:* :func:`mathkit.special_functions.systems.bessel.bessel_first_kind`
and :func:`~mathkit.special_functions.systems.bessel.bessel_second_kind`
wrap :func:`scipy.special.jv`/:func:`~scipy.special.yv` directly, and
are checked in this domain's tests against the defining differential
equation itself via finite differences.

*References:* F. W. Bessel, "Untersuchung des Theils der planetarischen
Storungen, welcher aus der Bewegung der Sonne entsteht," Abhandlungen
der Koniglichen Akademie der Wissenschaften zu Berlin (1824), 1-52.

.. minigallery:: ../../examples/special_functions/bessel/plot_01_bessel_functions.py

1782 -- 1859 -- Legendre and Chebyshev Polynomials
------------------------------------------------------

Adrien-Marie Legendre's 1782 work on the attraction of spheroids
introduced the polynomial family that solves Laplace's equation in
spherical coordinates and now bears his name, orthogonal on
:math:`[-1,1]` with the plainest possible weight, 1. Pafnuty Chebyshev's
1854 memoirs on least-deviation approximation (see the
numerical-analysis chronology) introduced a second family, orthogonal
with respect to the weight :math:`1/\sqrt{1-x^2}` and expressible in
closed form as :math:`T_n(x) = \cos(n\arccos x)` -- the family whose
minimal-oscillation property is exactly what makes Chebyshev *nodes* the
cure for Runge's phenomenon.

*Implementation:* :func:`mathkit.special_functions.systems.orthogonal_polynomials.legendre_polynomial`
and :func:`~mathkit.special_functions.systems.orthogonal_polynomials.chebyshev_polynomial`
wrap :func:`numpy.polynomial.legendre.legval`/:func:`~numpy.polynomial.chebyshev.chebval`
directly, with orthogonality verified numerically via
:func:`mathkit.special_functions.utils.orthogonality.inner_product`.

*References:* A.-M. Legendre, "Recherches sur l'attraction des
spheroides homogenes," Memoires de Mathematique et de Physique,
presentes a l'Academie Royale des Sciences 10 (1785), 411-434
(presented 1782).

.. minigallery:: ../../examples/special_functions/orthogonal_polynomials/plot_01_four_families.py

1810s -- 1879 -- Hermite and Laguerre Polynomials
-----------------------------------------------------

Pierre-Simon Laplace and, independently and more systematically,
Pafnuty Chebyshev studied polynomials orthogonal with respect to the
Gaussian weight :math:`e^{-x^2}` in the 1810s-1850s; Charles Hermite's
1864 memoir gave them their definitive treatment and the name they carry
today. Edmond Laguerre's 1879 paper introduced the complementary family
orthogonal on the half-line :math:`[0,\infty)` with weight
:math:`e^{-x}`. Both families would turn out, half a century later, to be
exactly the eigenfunctions quantum mechanics needed: Hermite polynomials
for the quantum harmonic oscillator, Laguerre polynomials for the radial
part of the hydrogen atom's wavefunctions.

*Implementation:* :func:`mathkit.special_functions.systems.orthogonal_polynomials.hermite_polynomial`
and :func:`~mathkit.special_functions.systems.orthogonal_polynomials.laguerre_polynomial`
wrap :func:`numpy.polynomial.hermite.hermval`/:func:`~numpy.polynomial.laguerre.lagval`,
with orthogonality against each family's own weight verified numerically
via :func:`scipy.integrate.quad` over an infinite domain.

*References:* C. Hermite, "Sur un nouveau developpement en serie des
fonctions," Comptes Rendus de l'Academie des Sciences 58 (1864), 93-100,
266-273; E. Laguerre, "Sur l'integrale
:math:`\int_x^\infty e^{-x}x^{-1}dx`," Bulletin de la Societe
Mathematique de France 7 (1879), 72-81.

.. minigallery:: ../../examples/special_functions/orthogonal_polynomials/plot_01_four_families.py

1822 -- 1965 -- Fourier's Theorem and the Fast Fourier Transform
-------------------------------------------------------------------

Joseph Fourier's 1822 treatise on heat conduction claimed, controversially
for its time, that essentially any periodic function can be represented
as a sum of sines and cosines -- a claim rigorously justified only
gradually over the following century, but one that gave the discrete
Fourier transform its entire conceptual foundation once digital
computation made it something to actually *compute*. James Cooley and
John Tukey's 1965 paper -- rediscovering, largely independently, an
algorithm Carl Friedrich Gauss had sketched already in an unpublished
1805 manuscript on interpolating asteroid orbits -- showed how to
compute the length-:math:`N` discrete Fourier transform recursively from
two length-:math:`N/2` transforms, reducing the cost from
:math:`O(N^2)` to :math:`O(N\log N)`: an algorithmic speedup so large
for the signal lengths radar and telecommunications needed that it is
routinely credited with making digital signal processing practical at
all.

.. math::

   X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi ikn/N}

*Implementation:* :func:`mathkit.special_functions.systems.fourier_transform.fft_numpy`
wraps :func:`numpy.fft.fft` as the primary API;
:func:`~mathkit.special_functions.systems.fourier_transform.dft_naive`
and :func:`~mathkit.special_functions.systems.fourier_transform.fft_radix2`
implement, hand-rolled, exactly the naive :math:`O(N^2)` sum and
Cooley-Tukey's radix-2 recursive halving respectively, with
:func:`~mathkit.special_functions.systems.fourier_transform.compare_fft_methods`
timing all three directly against each other to make the
:math:`O(N^2)` vs. :math:`O(N\log N)` gap concretely visible -- the one
sub-module in this domain where hand-rolling, rather than a direct
library call, is the point.

*References:* J. Fourier, *Theorie Analytique de la Chaleur* (Paris:
Firmin Didot, 1822); J. W. Cooley and J. W. Tukey, "An Algorithm for the
Machine Calculation of Complex Fourier Series," Mathematics of
Computation 19(90) (1965), 297-301; M. T. Heideman, D. H. Johnson, and
C. S. Burrus, "Gauss and the History of the Fast Fourier Transform,"
Archive for History of Exact Sciences 34(3) (1985), 265-277 (documenting
Gauss's unpublished 1805 anticipation).

.. minigallery:: ../../examples/special_functions/fourier_transform/plot_01_naive_vs_radix2_vs_numpy.py

See Also
--------

- :doc:`/api/special_functions`
- :doc:`/history/numerical_analysis_breakthroughs`
- :doc:`/history/calculus_breakthroughs`
