"""Abstract base classes and result containers for mathematicskit.fractals_chaos.

:class:`IteratedFunctionSystem` is the shared interface for the affine
random-IFS fractals (Barnsley fern, Sierpinski triangle/carpet) in
:mod:`mathematicskit.fractals_chaos.systems.ifs`; :class:`CellularAutomaton`
is the shared interface for the 1D/2D cellular automata in
:mod:`mathematicskit.fractals_chaos.systems.cellular_automata`. Both mirror
physicskit's one-ABC-per-algorithmic-family pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

__all__ = ["EscapeTimeResult", "BoxCountingResult", "IteratedFunctionSystem", "CellularAutomaton"]


@dataclass
class EscapeTimeResult:
    """Container for a Mandelbrot/Julia escape-time grid."""

    iterations: np.ndarray
    """ndarray, shape (ny, nx), int: Escape iteration count at each pixel
    (equal to ``max_iter`` for points that never escaped)."""

    extent: tuple = (-2.0, 1.0, -1.5, 1.5)
    """tuple: ``(re_min, re_max, im_min, im_max)`` plotted region."""

    max_iter: int = 100
    """int: Iteration cap used."""


@dataclass
class BoxCountingResult:
    """Container for a box-counting fractal-dimension estimate."""

    box_sizes: np.ndarray
    """ndarray: Box edge lengths used."""

    box_counts: np.ndarray
    """ndarray, int: Number of occupied boxes at each size."""

    dimension: float = 0.0
    """float: Estimated box-counting dimension (slope of
    ``log(count)`` vs. ``log(1/size)``)."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics slot (e.g. the linear fit's residuals)."""


class IteratedFunctionSystem(ABC):
    """Common base for random (chaos-game) iterated function systems.

    Concrete subclasses set ``self.transforms`` (a list of ``(A, b, p)``
    affine-map / probability triples, ``x -> A x + b`` chosen with
    probability ``p``) in ``__init__``.
    """

    transforms: list = []

    @abstractmethod
    def generate(self, n_points: int, seed: int = 0) -> np.ndarray:
        """Run the chaos game for ``n_points`` iterations.

        Parameters
        ----------
        n_points : int
        seed : int

        Returns
        -------
        ndarray, shape (n_points, 2)
        """


class CellularAutomaton(ABC):
    """Common base for cellular automata (1D elementary or 2D Game of Life)."""

    state: np.ndarray

    @abstractmethod
    def step(self) -> np.ndarray:
        """Advance the automaton by one generation and return the new state."""

    def run(self, n_steps: int) -> np.ndarray:
        """Run for ``n_steps`` generations, recording every state.

        Parameters
        ----------
        n_steps : int

        Returns
        -------
        ndarray, shape (n_steps + 1, ...)
            The initial state followed by each generation.
        """
        history = [self.state.copy()]
        for _ in range(n_steps):
            history.append(self.step().copy())
        return np.array(history)
