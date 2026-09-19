"""mathematicskit.fractals_chaos: discrete dynamical systems and fractal geometry.

The logistic map's Feigenbaum bifurcation cascade lives in
:mod:`mathematicskit.ode_dynamics` (:class:`~mathematicskit.ode_dynamics.systems.logistic_map.LogisticMap`,
:func:`~mathematicskit.ode_dynamics.systems.logistic_map.bifurcation_diagram`);
this domain covers what's distinctly *fractal*: Lyapunov exponent
estimation for 1D maps and flows (quantifying the chaos that cascade
ends in); fractal (box-counting) dimension estimation; Mandelbrot and
Julia set generation; iterated function systems (Barnsley fern,
Sierpinski triangle/carpet); and elementary (Wolfram-numbered) cellular
automata plus Conway's Game of Life.
"""

from mathematicskit.fractals_chaos.core.base import BoxCountingResult, CellularAutomaton, EscapeTimeResult, IteratedFunctionSystem
from mathematicskit.fractals_chaos.systems.box_counting import box_counting_dimension
from mathematicskit.fractals_chaos.systems.cellular_automata import ElementaryCA, GameOfLife
from mathematicskit.fractals_chaos.systems.ifs import BarnsleyFern, SierpinskiCarpet, SierpinskiTriangle
from mathematicskit.fractals_chaos.systems.lyapunov import lyapunov_exponent_1d_map, lyapunov_exponent_flow
from mathematicskit.fractals_chaos.systems.mandelbrot_julia import julia_set, mandelbrot_set
from mathematicskit.fractals_chaos.utils.ifs_utils import chaos_game

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "EscapeTimeResult",
    "BoxCountingResult",
    "IteratedFunctionSystem",
    "CellularAutomaton",
    "lyapunov_exponent_1d_map",
    "lyapunov_exponent_flow",
    "box_counting_dimension",
    "mandelbrot_set",
    "julia_set",
    "BarnsleyFern",
    "SierpinskiTriangle",
    "SierpinskiCarpet",
    "chaos_game",
    "ElementaryCA",
    "GameOfLife",
]
