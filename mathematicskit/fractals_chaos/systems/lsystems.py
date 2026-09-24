r"""Lindenmayer systems: parallel string rewriting and turtle-graphics
rendering.

Hand-rolled (no numpy/scipy equivalent). See A. Lindenmayer,
"Mathematical Models for Cellular Interactions in Development, I and
II," Journal of Theoretical Biology 18(3) (1968), 280-315, and P.
Prusinkiewicz and A. Lindenmayer, *The Algorithmic Beauty of Plants*
(New York: Springer, 1990), Ch. 1.
"""

from __future__ import annotations

import numpy as np

__all__ = ["lsystem", "turtle_path"]


def lsystem(axiom: str, rules: dict, iterations: int) -> str:
    r"""Rewrite every symbol of ``axiom`` in parallel, ``iterations`` times.

    Symbols without a rule are copied unchanged.

    Parameters
    ----------
    axiom : str
    rules : dict
        ``{symbol: replacement}``.
    iterations : int

    Returns
    -------
    str

    Examples
    --------
    >>> lsystem("A", {"A": "AB", "B": "A"}, 5)  # Lindenmayer's algae: lengths are Fibonacci numbers
    'ABAABABAABAAB'
    """
    word = axiom
    for _ in range(iterations):
        word = "".join(rules.get(symbol, symbol) for symbol in word)
    return word


def turtle_path(commands: str, angle: float, step: float = 1.0, heading: float = 90.0, draw: str = "FG") -> list:
    r"""Interpret an L-system word as turtle-graphics moves.

    ``F`` and ``G`` (or the symbols in ``draw``) move forward drawing a
    line, ``f`` moves without drawing, ``+``/``-`` turn left/right by
    ``angle`` degrees, and ``[``/``]`` push and pop the turtle's state
    (for branching plants). Other symbols are ignored.

    Parameters
    ----------
    commands : str
    angle : float
        Turning angle in degrees.
    step : float
    heading : float
        Initial heading in degrees (90 = up).
    draw : str
        Symbols that draw a forward line.

    Returns
    -------
    list of ndarray
        Polylines, each of shape (k, 2); a new polyline starts after each
        pen-up move or state pop.

    Examples
    --------
    >>> [line.round(6).tolist() for line in turtle_path("F+F", 90, heading=0)]
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]
    """
    x, y, theta = 0.0, 0.0, np.radians(heading)
    turn = np.radians(angle)
    stack = []
    lines, current = [], [(x, y)]
    for symbol in commands:
        if symbol in draw or symbol == "f":
            x, y = x + step * np.cos(theta), y + step * np.sin(theta)
            if symbol == "f":
                if len(current) > 1:
                    lines.append(np.array(current))
                current = [(x, y)]
            else:
                current.append((x, y))
        elif symbol == "+":
            theta += turn
        elif symbol == "-":
            theta -= turn
        elif symbol == "[":
            stack.append((x, y, theta))
        elif symbol == "]":
            if len(current) > 1:
                lines.append(np.array(current))
            x, y, theta = stack.pop()
            current = [(x, y)]
    if len(current) > 1:
        lines.append(np.array(current))
    return lines
