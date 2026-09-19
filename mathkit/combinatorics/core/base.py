r"""A Young/Ferrers diagram: the shared representation for
mathkit.combinatorics's integer-partition tools.

Unlike most other mathkit domains, this domain's algorithm families
(counting, partitions, inclusion-exclusion, special numbers) don't share
a common interface the way, e.g., :mod:`mathkit.numerical_analysis`'s
root finders do, so there's no ABC here -- only the one small class with
real behavior (rendering, conjugation) worth sharing, following
physicskit's own precedent of varying internal shape by what a family of
algorithms actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["YoungDiagram"]


@dataclass
class YoungDiagram:
    r"""A Young diagram for an integer partition :math:`\lambda = (\lambda_1 \geq \lambda_2 \geq \dots)`.

    Rows are left-justified, of non-increasing length -- the standard
    combinatorial picture of a partition, e.g. :math:`\lambda=(4,2,1)`
    of 7 is drawn as three rows of 4, 2, and 1 boxes. See Andrews &
    Eriksson, *Integer Partitions*, 2nd ed., Ch. 1.

    Parameters
    ----------
    parts : list of int
        Non-increasing positive integers summing to the partitioned number.

    Examples
    --------
    >>> diagram = YoungDiagram([4, 2, 1])
    >>> print(diagram.ferrers_diagram())
    ****
    **
    *
    >>> diagram.conjugate().parts
    [3, 2, 1, 1]
    """

    parts: list = field(default_factory=list)

    def __post_init__(self):
        if list(self.parts) != sorted(self.parts, reverse=True):
            raise ValueError("parts must be in non-increasing order")
        if any(p <= 0 for p in self.parts):
            raise ValueError("all parts must be positive")

    def ferrers_diagram(self, symbol: str = "*") -> str:
        """Render the diagram as a multi-line string of `symbol` characters.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        str
        """
        return "\n".join(symbol * p for p in self.parts)

    def conjugate(self) -> YoungDiagram:
        r"""The conjugate (transpose) partition: reflect the diagram across its main diagonal.

        :math:`\lambda'_k` = the number of parts of :math:`\lambda` that
        are :math:`\geq k`.

        Returns
        -------
        YoungDiagram
        """
        if not self.parts:
            return YoungDiagram([])
        max_part = self.parts[0]
        conjugate_parts = [sum(1 for p in self.parts if p >= k) for k in range(1, max_part + 1)]
        return YoungDiagram(conjugate_parts)

    @property
    def n(self) -> int:
        """int: The partitioned integer, ``sum(parts)``."""
        return sum(self.parts)
