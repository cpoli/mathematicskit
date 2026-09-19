r"""Elementary (1D) cellular automata with Wolfram rule numbering, and
Conway's Game of Life as a 2D cellular automaton.

See Wolfram, *A New Kind of Science*, 2002, Ch. 2-3, for elementary CA
and their rule numbering, and Gardner (1970), "The fantastic
combinations of John Conway's new solitaire game 'life'", Scientific
American 223, for Life's original presentation. Both boundary conditions
are periodic (toroidal).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mathkit.fractals_chaos.core.base import CellularAutomaton

__all__ = ["ElementaryCA", "GameOfLife"]


class ElementaryCA(CellularAutomaton):
    r"""1D elementary cellular automaton, Wolfram rule numbering (0-255).

    Each cell's next state is a function of its own and its two
    neighbors' current states -- 8 possible 3-cell neighborhoods, so
    :math:`2^8 = 256` possible rules, each identified by the 8-bit number
    whose bits give the output for each neighborhood (MSB-first:
    ``111, 110, 101, 100, 011, 010, 001, 000``), Wolfram's numbering
    convention. See Wolfram, *A New Kind of Science*, 2002, Ch. 2.

    Parameters
    ----------
    rule : int
        Wolfram rule number, ``0 <= rule <= 255``.
    width : int
        Number of cells (periodic boundary).
    initial : ndarray, shape (width,), optional
        Initial state (0s and 1s); defaults to a single ``1`` at the
        center cell, the standard way to display a rule's characteristic
        pattern.

    Examples
    --------
    >>> import numpy as np
    >>> # Rule 90 is the XOR of a cell's two neighbors, producing a
    >>> # discrete Sierpinski triangle from a single seed cell.
    >>> ca = ElementaryCA(rule=90, width=7)
    >>> ca.state.tolist()
    [0, 0, 0, 1, 0, 0, 0]
    >>> ca.step().tolist()
    [0, 0, 1, 0, 1, 0, 0]
    """

    def __init__(self, rule: int, width: int, initial=None):
        if not (0 <= rule <= 255):
            raise ValueError("rule must be between 0 and 255")
        self.rule = int(rule)
        self.width = int(width)
        # Bit k of `rule` gives the output for the neighborhood whose
        # 3-bit (left, self, right) pattern equals k.
        self._rule_table = np.array([(self.rule >> k) & 1 for k in range(8)], dtype=np.int64)
        if initial is None:
            state = np.zeros(self.width, dtype=np.int64)
            state[self.width // 2] = 1
        else:
            state = np.asarray(initial, dtype=np.int64).copy()
            if state.shape != (self.width,):
                raise ValueError("initial must have shape (width,)")
        self.state = state

    def step(self) -> np.ndarray:
        left = np.roll(self.state, 1)
        right = np.roll(self.state, -1)
        neighborhood = left * 4 + self.state * 2 + right
        self.state = self._rule_table[neighborhood]
        return self.state


@njit(cache=True)
def _life_step(state):
    ny, nx = state.shape
    new = np.empty_like(state)
    for i in range(ny):
        for j in range(nx):
            count = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    count += state[(i + di) % ny, (j + dj) % nx]
            if state[i, j] == 1:
                new[i, j] = 1 if (count == 2 or count == 3) else 0
            else:
                new[i, j] = 1 if count == 3 else 0
    return new


class GameOfLife(CellularAutomaton):
    r"""Conway's Game of Life: a 2D outer-totalistic cellular automaton (rule B3/S23).

    A dead cell with exactly 3 live neighbors becomes alive ("birth"); a
    live cell with 2 or 3 live neighbors survives, otherwise it dies
    ("death" by isolation or overcrowding). Periodic (toroidal) boundary.
    The per-cell neighbor count uses a Numba-compiled kernel
    (:func:`_life_step`), following physicskit's factory/dispatcher
    pattern for performance-critical stepping. See Gardner (1970),
    Scientific American 223.

    Parameters
    ----------
    initial : ndarray, shape (ny, nx)
        Initial grid (0s and 1s).

    Examples
    --------
    >>> import numpy as np
    >>> # A "blinker": a 3-cell line, period-2 oscillator.
    >>> grid = np.zeros((5, 5), dtype=np.int64)
    >>> grid[2, 1:4] = 1
    >>> life = GameOfLife(grid)
    >>> life.step().tolist() == [[0]*5, [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0]*5]
    True
    >>> np.array_equal(life.step(), grid)
    True
    """

    def __init__(self, initial):
        state = np.asarray(initial, dtype=np.int64)
        if state.ndim != 2:
            raise ValueError("initial must be a 2D grid")
        self.state = state

    def step(self) -> np.ndarray:
        self.state = _life_step(self.state)
        return self.state
