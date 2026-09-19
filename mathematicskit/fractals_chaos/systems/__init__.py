"""Concrete fractal-generation and discrete-dynamics algorithms."""

from mathematicskit.fractals_chaos.systems.box_counting import box_counting_dimension
from mathematicskit.fractals_chaos.systems.cellular_automata import ElementaryCA, GameOfLife
from mathematicskit.fractals_chaos.systems.ifs import BarnsleyFern, SierpinskiCarpet, SierpinskiTriangle
from mathematicskit.fractals_chaos.systems.lyapunov import lyapunov_exponent_1d_map, lyapunov_exponent_flow
from mathematicskit.fractals_chaos.systems.mandelbrot_julia import julia_set, mandelbrot_set

__all__ = [
    "box_counting_dimension",
    "ElementaryCA",
    "GameOfLife",
    "BarnsleyFern",
    "SierpinskiTriangle",
    "SierpinskiCarpet",
    "lyapunov_exponent_1d_map",
    "lyapunov_exponent_flow",
    "mandelbrot_set",
    "julia_set",
]
