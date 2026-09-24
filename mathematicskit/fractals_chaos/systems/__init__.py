"""Concrete fractal-generation and discrete-dynamics algorithms."""

from mathematicskit.fractals_chaos.systems.box_counting import box_counting_dimension
from mathematicskit.fractals_chaos.systems.cellular_automata import ElementaryCA, GameOfLife
from mathematicskit.fractals_chaos.systems.curves import divider_length, hilbert_curve, koch_curve, koch_snowflake, similarity_dimension, weierstrass_function
from mathematicskit.fractals_chaos.systems.dla import dla_cluster
from mathematicskit.fractals_chaos.systems.ifs import BarnsleyFern, SierpinskiCarpet, SierpinskiTriangle
from mathematicskit.fractals_chaos.systems.lsystems import lsystem, turtle_path
from mathematicskit.fractals_chaos.systems.lyapunov import lyapunov_exponent_1d_map, lyapunov_exponent_flow
from mathematicskit.fractals_chaos.systems.mandelbrot_julia import julia_set, mandelbrot_set
from mathematicskit.fractals_chaos.systems.maps import henon_map

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
    "divider_length",
    "hilbert_curve",
    "koch_curve",
    "koch_snowflake",
    "similarity_dimension",
    "weierstrass_function",
    "dla_cluster",
    "lsystem",
    "turtle_path",
    "henon_map",
]
