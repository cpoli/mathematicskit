Breakthroughs in Fractals and Chaos
===================================


.. include:: /_generated/nav/fractals_chaos.rst

.. epigraph::

   "Clouds are not spheres, mountains are not cones, coastlines are not
   circles."
   -- Benoit Mandelbrot, *The Fractal Geometry of Nature*, 1982

The objects in :mod:`mathematicskit.fractals_chaos` -- self-similar sets
of non-integer dimension, and cellular automata that build astonishing
complexity from a simple bit-flip rule -- share a common thread. Each
was dismissed at its discovery as a pathological curiosity, and each
later turned out to describe something real. This chronology traces
that thread from a 19th-century "monster" set to the computer-generated
images that gave the field its modern name.

.. contents:: Timeline
   :local:
   :depth: 1

1883 -- Cantor's Set and the First "Monster"
--------------------------------------------

Georg Cantor's 1883 construction starts from :math:`[0,1]` and
repeatedly removes the open middle third of every remaining interval.
The result is a set that is uncountably infinite, has total length
zero, and contains no interval at all. Nineteenth-century analysts
mostly treated such constructions as pathological counterexamples to be
avoided rather than objects worth studying. Nearly a century passed
before Benoit Mandelbrot's reframing (below) made "monster" sets like
Cantor's the central objects of an entire field.

*Connection:* the same self-similar, remove-and-recurse logic is the
subject matter of
:func:`mathematicskit.fractals_chaos.systems.box_counting.box_counting_dimension`.
The Cantor set's box-counting dimension, :math:`\log 2/\log 3 \approx
0.631`, is the standard first example of a non-integer fractal
dimension.

*References:* G. Cantor, "Über unendliche, lineare
Punktmannichfaltigkeiten (5)," Mathematische Annalen 21 (1883),
545-591; G. Cantor, "De la puissance des ensembles parfaits de points,"
Acta Mathematica 4 (1884), 381-392.

.. minigallery:: ../../examples/fractals_chaos/box_counting/plot_01_sierpinski_dimension.py

1890-1916 -- Peano and Sierpiński's Space-Filling and Gasket Curves
-------------------------------------------------------------------

Giuseppe Peano's 1890 continuous curve passes through every point of a
filled square. It shocked contemporaries who assumed that a "curve"
must be one-dimensional. Wacław Sierpiński's 1915 triangle, and his
closely related 1916 carpet, went the other way: they start from a
filled shape and repeatedly remove the middle triangle (or square) of
what remains. The limit is a self-similar set whose dimension is a
definite fraction between 1 and 2.

.. math::

   D = \frac{\log 3}{\log 2} \approx 1.585 \quad \text{(Sierpiński triangle)}

*Implementation:* :class:`mathematicskit.fractals_chaos.systems.ifs.SierpinskiTriangle`
and :class:`~mathematicskit.fractals_chaos.systems.ifs.SierpinskiCarpet`
generate both sets with the chaos game, and
:func:`mathematicskit.fractals_chaos.systems.box_counting.box_counting_dimension`
recovers their dimensions, :math:`\log 3/\log 2` (triangle) and
:math:`\log 8/\log 3` (carpet), numerically.

*References:* G. Peano, "Sur une courbe, qui remplit toute une aire
plane," Mathematische Annalen 36 (1890), 157-160; W. Sierpiński, "Sur
une courbe dont tout point est un point de ramification," Comptes
Rendus de l'Académie des Sciences Paris 160 (1915), 302-305.

.. minigallery:: ../../examples/fractals_chaos/ifs/plot_01_fern_and_sierpinski.py

.. minigallery:: ../../examples/fractals_chaos/box_counting/plot_01_sierpinski_dimension.py

1892-1985 -- Lyapunov Exponents: Measuring Sensitive Dependence
---------------------------------------------------------------

Edward Lorenz's discovery of sensitive dependence on initial conditions
(see the ODE-dynamics chronology) raised an obvious question: can that
sensitivity be *measured*, not just observed? The Lyapunov exponent
answers it. The exponent is named for Aleksandr Lyapunov's 1892 theory
of the stability of motion, but its modern algorithmic form, an average
logarithmic rate of separation, comes from the numerical-chaos
literature of the 1970s and 1980s. A positive exponent means that nearby
trajectories separate exponentially fast (chaos); a negative one means
that they converge (a stable orbit).

.. math::

   \lambda = \lim_{n\to\infty} \frac{1}{n}\sum_{k=0}^{n-1}\ln|f'(x_k)|

*Implementation:* :func:`mathematicskit.fractals_chaos.systems.lyapunov.lyapunov_exponent_1d_map`
and :func:`~mathematicskit.fractals_chaos.systems.lyapunov.lyapunov_exponent_flow`
compute this exponent for a 1D map and for a flow. The flow version
uses the shadow-trajectory renormalization method of Giancarlo Benettin
and colleagues (1980).

*References:* A. M. Lyapunov, *The General Problem of the Stability of
Motion* (Kharkov, 1892; in Russian); A. Wolf, J. B. Swift, H. L.
Swinney, and J. A. Vastano, "Determining Lyapunov Exponents from a Time
Series," Physica D 16(3) (1985), 285-317.

.. minigallery:: ../../examples/fractals_chaos/lyapunov/plot_01_logistic_map_route_to_chaos.py

1918-1919 -- Fatou and Julia's Iteration Theory
-----------------------------------------------

Around the end of the First World War, Pierre Fatou and Gaston Julia
independently developed a deep theory of what happens when a complex
rational function is applied to itself over and over. The complex plane
splits into a "Fatou set," where nearby orbits behave predictably, and a
complementary "Julia set," usually an intricate fractal boundary, where
they do not. Both worked without computers, using pure complex analysis
to describe sets that nobody could actually see for another sixty
years.

*Implementation:* :func:`mathematicskit.fractals_chaos.systems.mandelbrot_julia.julia_set`
computes the Julia set of :math:`z\mapsto z^2+c` by escape-time
iteration.

*References:* G. Julia, "Mémoire sur l'itération des fonctions
rationnelles," Journal de Mathématiques Pures et Appliquées, 8th series,
1 (1918), 47-245; P. Fatou, "Sur les équations fonctionnelles," Bulletin
de la Société Mathématique de France 47 (1919), 161-271.

.. minigallery:: ../../examples/fractals_chaos/mandelbrot_julia/plot_01_mandelbrot_and_julia.py

1970-1983 -- Conway's Life and Wolfram's Cellular Automata
----------------------------------------------------------

John Horton Conway devised the "Game of Life" in 1970, and Martin
Gardner's *Scientific American* column made it famous the same year.
It is a simple two-state birth-and-survival rule on a 2D grid, yet it
produces gliders and oscillators, and was later proved able to simulate
a universal Turing machine. Stephen Wolfram's systematic 1983 study
looked at the far simpler one-dimensional, two-color "elementary"
cellular automata. There are only 256 of them, each indexed by an 8-bit
rule number, yet they include rules that generate the Sierpiński
triangle (rule 90) and behavior Wolfram classified as chaotic
(rule 30).

*Implementation:* :class:`mathematicskit.fractals_chaos.systems.cellular_automata.GameOfLife`
and :class:`~mathematicskit.fractals_chaos.systems.cellular_automata.ElementaryCA`
implement both. The latter uses Wolfram's rule-numbering convention.

*References:* M. Gardner, "Mathematical Games: The Fantastic
Combinations of John Conway's New Solitaire Game 'Life'," Scientific
American 223(4) (1970), 120-123; S. Wolfram, "Statistical Mechanics of
Cellular Automata," Reviews of Modern Physics 55(3) (1983), 601-644.

.. minigallery:: ../../examples/fractals_chaos/cellular_automata/plot_01_wolfram_rules_and_life.py

1975 -- Mandelbrot Names "Fractal"
----------------------------------

Benoit Mandelbrot's 1975 book *Les objets fractals: forme, hasard et
dimension* coined the word "fractal," from the Latin *fractus*
("broken"). More importantly, it reframed the pathological curiosities
of Georg Cantor, Giuseppe Peano, and Gaston Julia as the right
mathematical language for rough, self-similar shapes -- coastlines,
mountain ranges, lung bronchi -- that smooth Euclidean geometry had
always struggled to model. Mandelbrot's 1980 computer image of what is
now called the Mandelbrot set became one of the most recognizable
images in mathematics. The set contains every complex :math:`c` whose
Julia set is connected.

*Implementation:* :func:`mathematicskit.fractals_chaos.systems.mandelbrot_julia.mandelbrot_set`
generates this set by Numba-accelerated escape-time iteration.

*References:* B. B. Mandelbrot, *Les objets fractals: forme, hasard et
dimension* (Paris: Flammarion, 1975); B. B. Mandelbrot, *The Fractal
Geometry of Nature* (San Francisco: W. H. Freeman, 1982).

.. minigallery:: ../../examples/fractals_chaos/mandelbrot_julia/plot_01_mandelbrot_and_julia.py

1988 -- Barnsley's Iterated Function Systems
--------------------------------------------

Michael Barnsley's 1988 book *Fractals Everywhere* built on John
Hutchinson's 1981 theory of self-similar sets. It gave a unified way to
generate self-similar fractals from a small set of contractive affine
maps applied in random order, a procedure called the "chaos game." One
image made the technique famous: a strikingly lifelike fern generated
from just four affine maps and their carefully chosen probabilities.

*Implementation:* :class:`mathematicskit.fractals_chaos.systems.ifs.BarnsleyFern`
implements Barnsley's four-map fern with
:func:`mathematicskit.fractals_chaos.utils.ifs_utils.chaos_game`.

*References:* J. E. Hutchinson, "Fractals and Self-Similarity," Indiana
University Mathematics Journal 30(5) (1981), 713-747; M. F. Barnsley,
*Fractals Everywhere* (Boston: Academic Press, 1988), Ch. 3.

.. minigallery:: ../../examples/fractals_chaos/ifs/plot_01_fern_and_sierpinski.py

See Also
--------

- :doc:`/api/fractals_chaos`
- :doc:`/history/ode_dynamics_breakthroughs`
- :doc:`/history/geometry_breakthroughs`
